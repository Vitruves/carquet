"""Writing Parquet files: :class:`ParquetWriter` and :func:`write_table`.

Data reaches the C writer through the Arrow C Data Interface
(``carquet_writer_write_arrow``), one RecordBatch at a time with no Python-side
copy; encoding, compression and I/O run in C without the GIL (pages are
encoded in parallel and row groups are written on a background thread).
"""

from __future__ import annotations

import base64
import ctypes
import os
from ctypes import addressof, byref
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional, Sequence, Union

from . import _ffi
from ._arrow import ArrowSchema
from ._types import (
    ENCODING_NAMES,
    RLE_DICTIONARY,
    codec_id,
    encoding_id,
    time_unit_id,
)
from .errors import CarquetError, StatusCode
from .schema import ParquetSchema

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa

__all__ = ["ParquetWriter", "write_table"]

DEFAULT_ROW_GROUP_SIZE = 1 << 20  # rows, like pyarrow.parquet.write_table


def _pa():
    from .reader import _pa as get

    return get()


# --------------------------------------------------------------------------
# Input normalisation
# --------------------------------------------------------------------------
def _is_pandas(obj: Any) -> bool:
    return type(obj).__module__.startswith("pandas.") and hasattr(obj, "columns") and hasattr(obj, "dtypes")


def to_arrow(data: Any) -> tuple:
    """``(schema, iterator of RecordBatch)`` for any supported tabular input."""
    pa = _pa()
    if isinstance(data, pa.Table):
        return data.schema, iter(data.to_batches())
    if isinstance(data, pa.RecordBatch):
        return data.schema, iter([data])
    if isinstance(data, pa.RecordBatchReader):
        return data.schema, iter(data)
    if _is_pandas(data):
        t = pa.Table.from_pandas(data)
        return t.schema, iter(t.to_batches())
    if hasattr(data, "__arrow_c_stream__"):  # polars, duckdb, nanoarrow, ...
        if hasattr(pa.RecordBatchReader, "from_stream"):
            r = pa.RecordBatchReader.from_stream(data)
            return r.schema, iter(r)
        t = pa.table(data)
        return t.schema, iter(t.to_batches())
    if hasattr(data, "__arrow_c_array__"):
        b = pa.record_batch(data)
        return b.schema, iter([b])
    if isinstance(data, Mapping):
        t = pa.table(data)
        return t.schema, iter(t.to_batches())
    if isinstance(data, (list, tuple)) or hasattr(data, "__next__"):
        it = iter(data)
        try:
            first = next(it)
        except StopIteration:
            raise ValueError("cannot infer a schema from an empty sequence of batches") from None
        if isinstance(first, pa.Table):
            first_batches = first.to_batches()
            rest = (b for t in it for b in t.to_batches())
            return first.schema, _chain(first_batches, rest)
        if not isinstance(first, pa.RecordBatch):
            raise TypeError(f"expected RecordBatch items, got {type(first).__name__}")
        return first.schema, _chain([first], it)
    raise TypeError(
        "expected a pyarrow Table / RecordBatch / RecordBatchReader, a pandas DataFrame, an object "
        f"implementing the Arrow PyCapsule interface, a dict of columns or a list of batches; got {type(data).__name__}"
    )


def _chain(first: list, rest: Iterable) -> Iterator:
    yield from first
    yield from rest


def _storage_type(pa, t):
    """The type carquet writes for Arrow type ``t``.

    Types without a Parquet mapping in the C bridge are written as their
    storage (durations as int64, view types as their plain counterparts, the
    null type as an all-null int32), like PyArrow does; the original type is
    kept in ``ARROW:schema`` and restored on read.
    """
    types = pa.types
    if types.is_duration(t):
        return pa.int64()
    if getattr(types, "is_string_view", None) and types.is_string_view(t):
        return pa.string()
    if getattr(types, "is_binary_view", None) and types.is_binary_view(t):
        return pa.binary()
    if types.is_null(t):
        return pa.int32()
    if types.is_struct(t):
        fields = [t.field(i) for i in range(t.num_fields)]
        new = [f.with_type(_storage_type(pa, f.type)) for f in fields]
        return t if new == fields else pa.struct(new)
    if types.is_map(t):
        k, v = _storage_type(pa, t.key_type), _storage_type(pa, t.item_type)
        if k == t.key_type and v == t.item_type:
            return t
        return pa.map_(t.key_field.with_type(k), t.item_field.with_type(v), keys_sorted=t.keys_sorted)
    if types.is_list(t) or types.is_large_list(t) or types.is_fixed_size_list(t):
        v = _storage_type(pa, t.value_type)
        if v == t.value_type:
            return t
        f = t.value_field.with_type(v)
        if types.is_fixed_size_list(t):
            return pa.list_(f, t.list_size)
        return pa.list_(f) if types.is_list(t) else pa.large_list(f)
    return t


def _storage_schema(pa, schema):
    fields = [f.with_type(_storage_type(pa, f.type)) for f in schema]
    if all(a.type == f.type for a, f in zip(fields, schema)):
        return None
    return pa.schema(fields, metadata=schema.metadata)


def _types_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    # nullability matters: the C writer rejects definition levels for a REQUIRED column
    return all(fa.name == fb.name and fa.type.equals(fb.type) and fa.nullable == fb.nullable
               for fa, fb in zip(a, b))


# --------------------------------------------------------------------------
# Option parsing
# --------------------------------------------------------------------------
def _format_version(version: Any) -> int:
    if version is None:
        return 2
    s = str(version)
    if s in ("1", "1.0"):
        return 1
    if s.startswith("2"):
        return 2
    raise ValueError(f"version must be '1.0' or '2.x', got {version!r}")


def _page_version(v: Any) -> int:
    s = str(v)
    if s in ("1", "1.0"):
        return 1
    if s in ("2", "2.0"):
        return 2
    raise ValueError(f"data_page_version must be 1 or 2 ('1.0' / '2.0'), got {v!r}")


class _Columns:
    """Resolves user column names to the writer's leaf column ordinals."""

    def __init__(self, schema: ParquetSchema) -> None:
        self.schema = schema

    def leaves(self, name: str) -> list:
        try:
            return [self.schema.column(name).index]
        except KeyError:
            pass
        hits = [c.index for c in self.schema.leaves_of(name)]
        if not hits:
            raise KeyError(f"column {name!r} not found (leaf columns: {self.schema.names})")
        return hits

    def select(self, spec: Union[bool, Sequence[str], None]) -> Optional[set]:
        """Leaf set for a ``bool | list[str]`` option (``None`` = option not used)."""
        if spec is None or spec is False:
            return None
        if spec is True:
            return {c.index for c in self.schema}
        if isinstance(spec, str):
            spec = [spec]
        out: set = set()
        for name in spec:
            out.update(self.leaves(name))
        return out


def _per_column(value: Any, columns: _Columns) -> tuple:
    """``(global_value, {leaf: value})`` for options that take a value or a ``{column: value}`` dict."""
    if isinstance(value, Mapping):
        per: dict = {}
        for name, v in value.items():
            for leaf in columns.leaves(name):
                per[leaf] = v
        return None, per
    return value, {}


def _sorting(spec: Any, columns: _Columns) -> list:
    out = []
    for item in spec:
        nulls_first = False
        if hasattr(item, "column_index") and hasattr(item, "descending"):  # pyarrow.parquet.SortingColumn
            idx, desc, nulls_first = int(item.column_index), bool(item.descending), bool(getattr(item, "nulls_first", False))
        else:
            if isinstance(item, str):
                name, order = item, "ascending"
            elif isinstance(item, (tuple, list)) and 2 <= len(item) <= 3:
                name, order = item[0], item[1]
                if len(item) == 3:
                    np = item[2]
                    nulls_first = np in ("at_start", "first", True)
            else:
                raise ValueError(f"sorting column must be a name or (name, 'ascending'|'descending'[, nulls]); got {item!r}")
            if isinstance(order, bool):
                desc = order
            elif str(order).lower() in ("ascending", "asc"):
                desc = False
            elif str(order).lower() in ("descending", "desc"):
                desc = True
            else:
                raise ValueError(f"sort order must be 'ascending' or 'descending', got {order!r}")
            leaves = columns.leaves(name)
            if len(leaves) != 1:
                raise ValueError(f"sorting column {name!r} must name a single leaf column")
            idx = leaves[0]
        out.append((idx, desc, nulls_first))
    return out


# --------------------------------------------------------------------------
# ParquetWriter
# --------------------------------------------------------------------------
class ParquetWriter:
    """Incremental Parquet writer.

    ``where`` is a path or a writable binary file object (the file is built in
    memory and written to it on :meth:`close`). Write any number of tables /
    batches; rows are packed into row groups of ``row_group_size`` rows
    regardless of how they were chunked on input::

        with carquet.ParquetWriter("out.parquet", table.schema, compression="zstd") as w:
            for batch in batches:
                w.write(batch)

    With ``append=True`` the file at ``where`` must exist; new row groups are
    added after the existing ones (``schema`` defaults to the file's schema).

    Options (all keyword-only; names follow ``pyarrow.parquet.write_table``):

    * ``compression``: ``"zstd"`` (default), ``"snappy"``, ``"gzip"``, ``"brotli"``,
      ``"lz4"`` (LZ4_RAW), ``"lz4_hadoop"``, ``"none"``; or ``{column: codec}``
      (unlisted columns are uncompressed). ``compression_level``: an int or a
      ``{column: level}`` dict.
    * ``row_group_size``: rows per row group (default 1 048 576).
    * ``data_page_size``: target page size in bytes, or ``{column: bytes}``.
    * ``max_rows_per_page``, ``write_batch_size``, ``dictionary_pagesize_limit``.
    * ``use_dictionary``: ``True`` / list of columns to dictionary-encode
      (default: carquet's PLAIN encoding). ``column_encoding``: an encoding for
      every column or ``{column: encoding}`` -- ``"plain"``, ``"dictionary"``,
      ``"delta_binary_packed"``, ``"delta_length_byte_array"``,
      ``"delta_byte_array"``, ``"byte_stream_split"``, ``"rle"``.
      ``use_byte_stream_split``: ``True`` (float columns) or a list of columns.
    * ``write_statistics``: ``True`` / ``False`` / list of columns;
      ``max_statistics_size`` caps stored min/max of binary columns (bytes).
    * ``write_page_index``: write the column / offset indexes (page pruning).
    * ``bloom_filters``: ``True``, a list of columns, or ``{column: {"ndv": n,
      "fpp": p}}``.
    * ``sorting_columns``: declared sort order -- names, ``(name, "descending")``
      tuples or ``pyarrow.parquet.SortingColumn`` objects (not verified).
    * ``coerce_timestamps`` (``"ms"`` / ``"us"`` / ``"ns"``) and
      ``allow_truncated_timestamps``.
    * ``data_page_version``: ``"1.0"`` (default) or ``"2.0"``; ``version``: the
      footer format version (``"2.6"`` default, ``"1.0"`` for very old readers).
    * ``write_page_checksum`` (alias ``write_crc``): page CRC32 (default on).
    * ``store_schema``: embed the Arrow schema (``ARROW:schema``) and the
      schema metadata (e.g. pandas) so readers restore exact Arrow types.
    * ``metadata``: extra ``{key: value}`` footer metadata.
    """

    def __init__(
        self,
        where: Any,
        schema: Any = None,
        *,
        append: bool = False,
        compression: Any = "zstd",
        compression_level: Any = None,
        row_group_size: Optional[int] = None,
        data_page_size: Any = None,
        data_page_version: Any = 1,
        version: Any = "2.6",
        use_dictionary: Any = None,
        column_encoding: Any = None,
        use_byte_stream_split: Any = None,
        write_statistics: Any = True,
        max_statistics_size: Optional[int] = None,
        write_page_index: bool = False,
        write_page_checksum: Optional[bool] = None,
        write_crc: Optional[bool] = None,
        bloom_filters: Any = None,
        sorting_columns: Optional[Sequence[Any]] = None,
        coerce_timestamps: Optional[str] = None,
        allow_truncated_timestamps: bool = False,
        write_batch_size: Optional[int] = None,
        max_rows_per_page: Optional[int] = None,
        dictionary_pagesize_limit: Optional[int] = None,
        store_schema: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        page_size: Any = None,
    ) -> None:
        pa = _pa()
        lib = _ffi.get_lib()
        self._writer: Optional[int] = None
        self._pa = pa
        self._sink = None
        self.path: Optional[str] = None
        self._created_file = False
        self.num_rows = 0
        self._rows_in_group = 0

        if _is_path_like(where):
            self.path = os.fsdecode(where)
        elif hasattr(where, "write"):
            if append:
                raise ValueError("append=True needs a path")
            self._sink = where
        else:
            raise TypeError(f"where must be a path or a writable binary file object, got {type(where).__name__}")

        if schema is None:
            if not append:
                raise TypeError("schema is required (or use write_table / append=True)")
            from .reader import ParquetFile

            with ParquetFile(self.path, memory_map=False) as pf:
                schema = pf.schema
        elif not isinstance(schema, pa.Schema):
            schema = pa.schema(schema)
        self.schema = schema
        self.row_group_size = int(row_group_size) if row_group_size else DEFAULT_ROW_GROUP_SIZE
        if self.row_group_size <= 0:
            raise ValueError("row_group_size must be a positive integer")
        if store_schema is None:
            store_schema = not append

        # Arrow schema -> carquet schema (the import consumes the ArrowSchema).
        self._storage = _storage_schema(pa, schema)
        c_schema = ArrowSchema()
        _export_schema(pa, self._storage or schema, c_schema)
        cq_schema = ctypes.c_void_p()
        err = _ffi.CarquetErrorStruct()
        _ffi.check(lib.carquet_arrow_import_schema(byref(c_schema), byref(cq_schema), byref(err)), err, "schema")
        try:
            self.parquet_schema = ParquetSchema._from_handle(cq_schema)
            cols = _Columns(self.parquet_schema)

            opts = _ffi.WriterOptions()
            lib.carquet_writer_options_init(byref(opts))
            codec, per_codec = _per_column(compression, cols)
            opts.compression = codec_id(codec) if codec is not None or not per_codec else 0
            level, per_level = _per_column(compression_level, cols)
            opts.compression_level = int(level) if level is not None else 0
            if page_size is not None and data_page_size is None:
                data_page_size = page_size
            psize, per_psize = _per_column(data_page_size, cols)
            if psize is not None:
                opts.page_size = int(psize)
            opts.data_page_version = _page_version(data_page_version)
            opts.file_format_version = _format_version(version)
            opts.write_statistics = write_statistics is not False
            opts.write_page_index = bool(write_page_index)
            crc = write_page_checksum if write_page_checksum is not None else write_crc
            if crc is not None:
                opts.write_crc = bool(crc)
            # ARROW:schema is serialised by PyArrow itself below (exact Arrow
            # types: time zones, dictionaries, large / extension types, nested
            # field names), so the C writer's own copy is turned off.
            opts.write_arrow_schema = False
            if coerce_timestamps is not None:
                opts.coerce_timestamps = True
                opts.coerce_timestamp_unit = time_unit_id(coerce_timestamps)
            opts.allow_timestamp_truncation = bool(allow_truncated_timestamps)
            if write_batch_size is not None:
                opts.write_batch_size = int(write_batch_size)
            if max_rows_per_page is not None:
                opts.max_rows_per_page = int(max_rows_per_page)
                # the C writer checks the row cap once per write batch: make
                # the batches no larger than the cap so pages hold exactly it
                if write_batch_size is None and int(max_rows_per_page) > 0:
                    opts.write_batch_size = int(max_rows_per_page)
            if dictionary_pagesize_limit is not None:
                opts.dictionary_page_size = int(dictionary_pagesize_limit)

            err = _ffi.CarquetErrorStruct()
            if self._sink is not None:
                writer = lib.carquet_writer_create_buffer(cq_schema, byref(opts), byref(err))
            elif append:
                writer = lib.carquet_writer_open_append(_ffi.fsencode(self.path), cq_schema, byref(opts), byref(err))
            else:
                writer = lib.carquet_writer_create(_ffi.fsencode(self.path), cq_schema, byref(opts), byref(err))
                self._created_file = bool(writer)
            if not writer:
                _ffi.check(err.code or StatusCode.FILE_OPEN, err, self.path or "<buffer>")
        finally:
            lib.carquet_schema_free(cq_schema)
        self._writer = writer

        try:
            self._configure_columns(cols, per_codec, codec, per_level, level, per_psize, use_dictionary,
                                    column_encoding, use_byte_stream_split, write_statistics,
                                    max_statistics_size, bloom_filters, sorting_columns)
            if store_schema:
                blob = base64.b64encode(schema.serialize().to_pybytes())
                self.add_metadata("ARROW:schema", blob)
            if store_schema and schema.metadata:
                for k, v in schema.metadata.items():
                    key = k.decode("utf-8", "replace")
                    if key != "ARROW:schema":
                        self.add_metadata(key, v.decode("utf-8", "replace"))
            for k, v in (metadata or {}).items():
                self.add_metadata(k, v)
        except BaseException:
            self._abort()
            raise

    def _configure_columns(self, cols, per_codec, codec, per_level, level, per_psize, use_dictionary,
                           column_encoding, use_byte_stream_split, write_statistics, max_statistics_size,
                           bloom_filters, sorting_columns) -> None:
        lib = _ffi.get_lib()
        w = self._writer

        def check(st, what):
            _ffi.check(st, None, what)

        # encodings: use_dictionary < use_byte_stream_split < column_encoding
        enc: dict = {}
        dict_cols = cols.select(use_dictionary)
        if dict_cols:
            for leaf in dict_cols:
                if use_dictionary is not True or cols.schema[leaf].physical_type != "BOOLEAN":
                    enc[leaf] = RLE_DICTIONARY
        if use_byte_stream_split:
            if use_byte_stream_split is True:
                bss = {c.index for c in cols.schema if c.physical_type in ("FLOAT", "DOUBLE")}
            else:
                bss = cols.select(use_byte_stream_split) or set()
            for leaf in bss:
                enc[leaf] = ENCODING_NAMES["byte_stream_split"]
        if column_encoding is not None:
            g, per = _per_column(column_encoding, cols)
            if g is not None:
                for c in cols.schema:
                    enc[c.index] = encoding_id(g)
            for leaf, e in per.items():
                enc[leaf] = encoding_id(e)
        for leaf, e in sorted(enc.items()):
            check(lib.carquet_writer_set_column_encoding(w, leaf, e), f"encoding of {cols.schema[leaf].path}")

        for leaf in sorted(set(per_codec) | set(per_level)):
            c = codec_id(per_codec[leaf]) if leaf in per_codec else codec_id(codec)
            lvl = per_level.get(leaf, level)
            check(lib.carquet_writer_set_column_compression(w, leaf, c, int(lvl) if lvl is not None else 0),
                  f"compression of {cols.schema[leaf].path}")
        for leaf, size in per_psize.items():
            check(lib.carquet_writer_set_column_page_size(w, leaf, int(size)), f"page size of {cols.schema[leaf].path}")

        if write_statistics is not True and write_statistics is not False:
            keep = cols.select(write_statistics) or set()
            for c in cols.schema:
                if c.index not in keep:
                    check(lib.carquet_writer_set_column_statistics(w, c.index, False), "statistics")
        if max_statistics_size is not None:
            check(lib.carquet_writer_set_max_statistics_size(w, int(max_statistics_size)), "max_statistics_size")

        if bloom_filters:
            if isinstance(bloom_filters, Mapping):
                items = []
                for name, spec in bloom_filters.items():
                    for leaf in cols.leaves(name):
                        items.append((leaf, spec))
            else:
                items = [(leaf, True) for leaf in sorted(cols.select(bloom_filters) or ())]
            for leaf, spec in items:
                if spec is False or spec is None:
                    continue
                ndv, fpp = 0, 0.0
                if isinstance(spec, Mapping):
                    ndv, fpp = int(spec.get("ndv", 0) or 0), float(spec.get("fpp", 0.0) or 0.0)
                elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                    ndv, fpp = int(spec[0] or 0), float(spec[1] or 0.0)
                check(lib.carquet_writer_set_column_bloom_filter_options(w, leaf, True, ndv, fpp), "bloom filter")

        if sorting_columns:
            entries = _sorting(sorting_columns, cols)
            arr = (_ffi.SortingColumnStruct * len(entries))()
            for i, (idx, desc, nf) in enumerate(entries):
                arr[i].column_index, arr[i].descending, arr[i].nulls_first = idx, desc, nf
            check(lib.carquet_writer_set_sorting_columns(w, arr, len(entries)), "sorting_columns")

    # -- writing -------------------------------------------------------------
    @property
    def closed(self) -> bool:
        return not self._writer

    def _require_open(self) -> int:
        if not self._writer:
            raise CarquetError(StatusCode.ALREADY_CLOSED, "writer is closed")
        return self._writer

    def add_metadata(self, key: str, value: Optional[str]) -> None:
        """Add a footer key/value entry (before :meth:`close`)."""
        w = self._require_open()
        k = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        if value is None:
            v = None
        elif isinstance(value, bytes):
            v = value
        else:
            v = str(value).encode("utf-8")
        _ffi.check(_ffi.get_lib().carquet_writer_add_metadata(w, k, v), None, f"metadata {key!r}")

    def write(self, data: Any, row_group_size: Optional[int] = None) -> None:
        """Write a table, record batch, batch reader, DataFrame or Arrow stream."""
        self._require_open()
        schema, batches = to_arrow(data)
        conform = None
        if not _types_equal(schema, self.schema):
            conform = self._conformer(schema)
        rg_size = int(row_group_size) if row_group_size else self.row_group_size
        for batch in batches:
            if conform is not None:
                batch = conform(batch)
            self._write_batch(batch, rg_size)

    write_table = write
    write_batch = write

    def _conformer(self, schema):
        pa = self._pa
        if [f.name for f in schema] != [f.name for f in self.schema]:
            raise ValueError(
                f"data columns {schema.names} do not match the writer schema {self.schema.names}"
            )

        def conform(batch):
            if batch.num_rows == 0:
                return None
            try:
                return pa.Table.from_batches([batch]).cast(self.schema).to_batches()[0]
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
                raise ValueError(f"data schema does not match the writer schema: {exc}") from None

        return conform

    def _write_batch(self, batch, rg_size: int) -> None:
        if batch is None or batch.num_rows == 0:
            return
        if self._storage is not None:
            batch = self._pa.Table.from_batches([batch]).cast(self._storage).to_batches()[0]
        start, n = 0, batch.num_rows
        while start < n:
            room = rg_size - self._rows_in_group
            take = min(room, n - start)
            part = batch if (start == 0 and take == n) else batch.slice(start, take)
            self._write_arrow(part)
            start += take
            self._rows_in_group += take
            self.num_rows += take
            if self._rows_in_group >= rg_size:
                self.flush()

    def _write_arrow(self, batch) -> None:
        from ._arrow import ArrowArray

        lib = _ffi.get_lib()
        array = ArrowArray()
        c_schema = ArrowSchema()
        batch._export_to_c(addressof(array), addressof(c_schema))
        err = _ffi.CarquetErrorStruct()
        st = lib.carquet_writer_write_arrow(self._writer, byref(array), byref(c_schema), byref(err))
        _ffi.check(st, err, "write")

    def flush(self) -> None:
        """End the current row group now (no-op when it is empty)."""
        w = self._require_open()
        if self._rows_in_group:
            _ffi.check(_ffi.get_lib().carquet_writer_new_row_group(w), None, "row group flush")
            self._rows_in_group = 0

    def close(self) -> None:
        """Finish the file (footer) and release the writer. Idempotent."""
        w, self._writer = self._writer, None
        if not w:
            return
        lib = _ffi.get_lib()
        st = lib.carquet_writer_close(w)
        if self._sink is None:
            _ffi.check(st, None, f"close {self.path}")
            return
        if st != _ffi.OK:
            lib.carquet_writer_abort(w)
            _ffi.check(st, None, "close")
        ptr = ctypes.c_void_p()
        size = ctypes.c_size_t()
        st = lib.carquet_writer_get_buffer(w, byref(ptr), byref(size))
        if st != _ffi.OK:
            lib.carquet_writer_abort(w)
            _ffi.check(st, None, "get buffer")
        try:
            view = (ctypes.c_char * size.value).from_address(ptr.value)
            self._sink.write(memoryview(view).cast("B"))
        finally:
            _ffi.free_buffer(ptr.value)

    def _abort(self) -> None:
        w, self._writer = self._writer, None
        if w:
            _ffi.get_lib().carquet_writer_abort(w)
        if self._created_file and self.path:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def abort(self) -> None:
        """Discard the file being written (a new file is removed; nothing is finalised)."""
        self._abort()

    def __enter__(self) -> "ParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()

    def __del__(self) -> None:  # pragma: no cover - best effort
        # like pyarrow.parquet.ParquetWriter: an unclosed writer is finished, not discarded
        try:
            if self._writer:
                self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        target = self.path or type(self._sink).__name__
        state = "closed" if self.closed else f"rows={self.num_rows}"
        return f"<carquet.ParquetWriter {target!r} {state}>"


def _is_path_like(obj: Any) -> bool:
    return isinstance(obj, (str, os.PathLike))


def _export_schema(pa, schema, c_schema) -> None:
    schema._export_to_c(addressof(c_schema))


def write_table(table: Any, where: Any, *, row_group_size: Optional[int] = None, **options) -> None:
    """Write a table (or any input :meth:`ParquetWriter.write` accepts) to ``where``.

    ``where`` is a path or a writable binary file object (``io.BytesIO``, an
    open file, ...). See :class:`ParquetWriter` for the options; with
    ``append=True`` the rows are added to an existing file. If writing fails,
    a file created by this call is removed.
    """
    schema, batches = to_arrow(table)
    if options.get("append"):
        schema = None  # the file's schema; the data is conformed to it
    w = ParquetWriter(where, schema, row_group_size=row_group_size, **options)
    try:
        for b in batches:
            w._write_batch(b if _types_equal(b.schema, w.schema) else w._conformer(b.schema)(b), w.row_group_size)
        w.close()
    except BaseException:
        w._abort()
        raise

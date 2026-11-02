"""Reading Parquet files: :class:`ParquetFile` and the ``read_*`` functions.

Two C read paths back the Arrow output:

* the **batch reader** (``carquet_batch_reader_*``) for flat columns and
  single-level ``list`` / ``map`` columns of primitives -- column projection,
  row-group selection, page-index pruning, row ranges and the library's
  parallel / pipelined decoding all happen in C; each batch is handed to
  PyArrow through the Arrow C Data Interface;
* the **nested assembler** (``carquet_reader_read_arrow_fields``) for structs
  and deeper nesting, one row group at a time, reading only the projected
  top-level fields.

Every C call runs without the GIL.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import weakref
from ctypes import addressof, byref, c_int32, c_void_p
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Sequence, Union

from . import _ffi
from ._arrow import ArrowArray, ArrowSchema, release_array, release_schema
from ._convert import Converter
from .errors import CarquetError, StatusCode
from .filters import Filters, RowGroupPruner, build_page_filter, normalize_filters
from .metadata import (
    BloomFilter,
    ColumnIndex,
    FileMetaData,
    OffsetIndex,
    RowGroupMetaData,
    read_key_value_metadata,
)
from .schema import ColumnSchema, ParquetSchema

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa

__all__ = [
    "FileInfo",
    "ParquetFile",
    "info",
    "iter_batches",
    "read_metadata",
    "read_pandas",
    "read_schema",
    "read_table",
    "validate",
]

PathLike = Union[str, "os.PathLike[str]"]
Source = Union[PathLike, bytes, bytearray, memoryview, Any]

DEFAULT_BATCH_SIZE = 65536
# Batch size used when the whole result is materialised (read / read_table):
# large enough that a row group usually comes back as a single chunk.
_READ_BATCH_SIZE = 1 << 20


def _pa():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
        raise ImportError(
            "pyarrow is required for this carquet function; install it with "
            "`pip install pyarrow` or `pip install 'carquet[arrow]'`"
        ) from exc
    return pa


def _import_batch(pa, array: ArrowArray, schema) -> "pa.RecordBatch":
    """Move a top-level struct ArrowArray into a RecordBatch of ``schema``."""
    try:
        return pa.RecordBatch._import_from_c(addressof(array), schema)
    except BaseException:
        release_array(array)
        raise


def _is_path(source: Any) -> bool:
    return isinstance(source, (str, os.PathLike))


# --------------------------------------------------------------------------
# Footer-only helpers (no PyArrow needed)
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FileInfo:
    """Footer-level facts about a Parquet file (``carquet_file_info_t``)."""

    path: str
    num_rows: int
    num_row_groups: int
    num_columns: int
    file_size: int
    version: int
    created_by: str


def info(path: PathLike) -> FileInfo:
    """Read only the footer of ``path`` and return a :class:`FileInfo`.

    Raises :class:`CarquetError` if the file is missing or not a Parquet file.
    """
    lib = _ffi.get_lib()
    out = _ffi.FileInfoStruct()
    err = _ffi.CarquetErrorStruct()
    st = lib.carquet_get_file_info(_ffi.fsencode(path), byref(out), byref(err))
    _ffi.check(st, err, str(os.fsdecode(path)))
    return FileInfo(
        path=os.fsdecode(path),
        num_rows=int(out.num_rows),
        num_row_groups=int(out.num_row_groups),
        num_columns=int(out.num_columns),
        file_size=int(out.file_size),
        version=int(out.version),
        created_by=out.created_by.decode("utf-8", "replace"),
    )


def validate(path: PathLike) -> None:
    """Structurally validate ``path`` (magic, footer, metadata); raise :class:`CarquetError` on failure."""
    lib = _ffi.get_lib()
    err = _ffi.CarquetErrorStruct()
    st = lib.carquet_validate_file(_ffi.fsencode(path), byref(err))
    _ffi.check(st, err, str(os.fsdecode(path)))


# --------------------------------------------------------------------------
# Projection planning
# --------------------------------------------------------------------------
class _Field:
    """A top-level field: its leaf columns and which C path can read it."""

    __slots__ = ("ordinal", "name", "leaves", "batchable")

    def __init__(self, ordinal: int, name: str, leaves: list, batchable: bool) -> None:
        self.ordinal = ordinal
        self.name = name
        self.leaves = leaves
        self.batchable = batchable


def _plan_fields(schema: ParquetSchema) -> list:
    groups = {e.name: e for e in schema.elements if e.depth == 1 and not e.is_leaf}
    fields: list = []
    by_name: dict = {}
    for col in schema:
        top = col.parts[0]
        f = by_name.get(top)
        if f is None:
            f = _Field(len(fields), top, [], False)
            by_name[top] = f
            fields.append(f)
        f.leaves.append(col)
    for f in fields:
        if len(f.leaves) == 1 and len(f.leaves[0].parts) == 1:
            f.batchable = f.leaves[0].max_repetition_level == 0
            continue
        g = groups.get(f.name)
        kind = g.logical_type.name if g is not None and g.logical_type is not None else None
        single = all(len(c.parts) == 3 and c.max_repetition_level == 1 for c in f.leaves)
        f.batchable = single and (
            (kind == "LIST" and len(f.leaves) == 1) or (kind == "MAP" and len(f.leaves) == 2)
        )
    return fields


# --------------------------------------------------------------------------
# ParquetFile
# --------------------------------------------------------------------------
class ParquetFile:
    """An open Parquet file.

    ``source`` is a path, a bytes-like object (``bytes``, ``bytearray``,
    ``memoryview``, ``pyarrow.Buffer``) or a binary file object; buffers are
    read in place, without copying. Use it as a context manager or call
    :meth:`close`::

        with carquet.ParquetFile("data.parquet") as pf:
            pf.schema                     # pyarrow.Schema
            pf.parquet_schema             # carquet.ParquetSchema (physical/logical types)
            pf.metadata.row_group(0).column("id").statistics
            pf.read(columns=["id", "name"], filters=[("id", ">", 100)])
            for batch in pf.iter_batches(batch_size=10_000):
                ...

    Args:
        memory_map: map the file instead of buffered reads (paths only).
        verify_checksums: verify page CRC32 checksums when present.
        num_threads: decoder threads (0 = one per core).
        buffer_size: read buffer size in bytes for buffered (non-mmap) I/O.
        restore_arrow_schema: apply the Arrow types stored in the file's
            ``ARROW:schema`` entry (time zones, large/dictionary types, field
            metadata...), as ``pyarrow.parquet`` does.

    A ``ParquetFile`` is not thread-safe; open one per thread.
    """

    def __init__(
        self,
        source: Source,
        *,
        memory_map: bool = True,
        verify_checksums: bool = True,
        num_threads: int = 0,
        buffer_size: Optional[int] = None,
        restore_arrow_schema: bool = True,
        use_mmap: Optional[bool] = None,
    ) -> None:
        if use_mmap is not None:  # 0.7.2 spelling
            memory_map = bool(use_mmap)
        self._handle: Optional[int] = None
        self._keepalive: Any = None
        self._buffer_size: Optional[int] = None
        self._streams: "weakref.WeakSet" = weakref.WeakSet()
        self._lock = threading.RLock()
        self._num_threads = int(num_threads)
        self._restore = bool(restore_arrow_schema)

        lib = _ffi.get_lib()
        opts = _ffi.ReaderOptions()
        lib.carquet_reader_options_init(byref(opts))
        opts.verify_checksums = bool(verify_checksums)
        opts.num_threads = self._num_threads
        if buffer_size is not None:
            opts.buffer_size = int(buffer_size)
        err = _ffi.CarquetErrorStruct()

        if _is_path(source):
            self.path: Optional[str] = os.fsdecode(source)
            opts.use_mmap = bool(memory_map)
            handle = lib.carquet_reader_open(_ffi.fsencode(source), byref(opts), byref(err))
        else:
            self.path = None
            data = source
            if hasattr(source, "getbuffer"):  # io.BytesIO: zero-copy view
                data = source.getbuffer()
            elif hasattr(source, "read") and not isinstance(source, (bytes, bytearray, memoryview)):
                data = source.read()
            try:
                addr, size, keep = _ffi.buffer_address(data)
            except TypeError:
                raise TypeError(
                    "source must be a path, a bytes-like object or a binary file object, "
                    f"got {type(source).__name__}"
                ) from None
            self._keepalive = keep
            self._buffer_size = size
            opts.use_mmap = False
            handle = lib.carquet_reader_open_buffer(addr or None, size, byref(opts), byref(err))
        if not handle:
            _ffi.check(err.code or StatusCode.FILE_OPEN, err, self.path or "<buffer>")
        self._handle = handle
        self._memory_map = bool(lib.carquet_reader_is_mmap(handle))
        self._schema_handle = lib.carquet_reader_schema(handle)

        self.parquet_schema: ParquetSchema = ParquetSchema._from_handle(self._schema_handle)
        self.num_rows: int = int(lib.carquet_reader_num_rows(handle))
        self.num_row_groups: int = int(lib.carquet_reader_num_row_groups(handle))
        self.num_columns: int = int(lib.carquet_reader_num_columns(handle))
        self._fields = _plan_fields(self.parquet_schema)
        self._field_by_name = {f.name: f for f in self._fields}
        self._raw_arrow_schema: Any = None
        self._metadata: Optional[FileMetaData] = None
        self._kv: Optional[dict] = None
        self._converters: dict = {}

    # -- lifecycle ---------------------------------------------------------
    @property
    def closed(self) -> bool:
        return not self._handle

    def _require_open(self) -> int:
        if not self._handle:
            raise CarquetError(StatusCode.ALREADY_CLOSED, f"{self._name}: file is closed")
        return self._handle

    @property
    def _name(self) -> str:
        return self.path or "<buffer>"

    def close(self) -> None:
        """Release the file (and stop any batch iterator still reading it)."""
        with self._lock:
            for s in list(self._streams):
                s.close(interrupted=True)
            h, self._handle = self._handle, None
            if h:
                _ffi.get_lib().carquet_reader_close(h)
            self._keepalive = None

    def __enter__(self) -> "ParquetFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self.closed:
            return f"<carquet.ParquetFile {self._name!r} closed>"
        return (
            f"<carquet.ParquetFile {self._name!r} rows={self.num_rows} "
            f"row_groups={self.num_row_groups} columns={self.num_columns}>"
        )

    # -- schema & metadata ---------------------------------------------------
    def _raw_schema(self):
        if self._raw_arrow_schema is None:
            handle = self._require_open()
            pa = _pa()
            lib = _ffi.get_lib()
            c_schema = ArrowSchema()
            err = _ffi.CarquetErrorStruct()
            st = lib.carquet_arrow_export_schema(lib.carquet_reader_schema(handle), byref(c_schema), byref(err))
            _ffi.check(st, err, self._name)
            try:
                self._raw_arrow_schema = pa.Schema._import_from_c(addressof(c_schema))
            except BaseException:
                release_schema(c_schema)
                raise
        return self._raw_arrow_schema

    @property
    def schema(self) -> "pa.Schema":
        """The file schema as a :class:`pyarrow.Schema` (what :meth:`read` returns)."""
        return self._converter([f.ordinal for f in self._fields]).schema

    @property
    def schema_arrow(self) -> "pa.Schema":
        """Alias of :attr:`schema` (``pyarrow.parquet.ParquetFile`` spelling)."""
        return self.schema

    @property
    def key_value_metadata(self) -> dict:
        """The footer's key/value metadata (``str`` -> ``str`` or ``None``)."""
        if self._kv is None:
            self._kv = read_key_value_metadata(self._require_open())
        return self._kv

    @property
    def metadata(self) -> FileMetaData:
        """Footer metadata: row groups, column chunks, statistics (a :class:`FileMetaData`)."""
        if self._metadata is None:
            handle = self._require_open()
            created_by = version = file_size = None
            if self.path is not None:
                try:
                    fi = info(self.path)
                    created_by, version, file_size = fi.created_by, fi.version, fi.file_size
                except CarquetError:
                    pass
            else:
                file_size = self._buffer_size
            self._metadata = FileMetaData(
                num_rows=self.num_rows,
                schema=self.parquet_schema,
                row_groups=[RowGroupMetaData._read(handle, i, self.parquet_schema) for i in range(self.num_row_groups)],
                key_value_metadata=dict(self.key_value_metadata),
                created_by=created_by,
                format_version=version,
                file_size=file_size,
            )
        return self._metadata

    def row_group(self, i: int) -> RowGroupMetaData:
        """Metadata of row group ``i`` (same as ``metadata.row_group(i)``)."""
        return self.metadata.row_group(i)

    def _column(self, column: Union[int, str]) -> ColumnSchema:
        return self.parquet_schema.column(column)

    def _check_rg(self, row_group: int) -> int:
        if not -self.num_row_groups <= row_group < self.num_row_groups:
            raise IndexError(f"row group {row_group} out of range (file has {self.num_row_groups})")
        return row_group % self.num_row_groups if self.num_row_groups else row_group

    def bloom_filter(self, column: Union[int, str], row_group: int = 0) -> Optional[BloomFilter]:
        """The bloom filter of ``column`` in ``row_group``, or ``None`` if none was written."""
        handle = self._require_open()
        col = self._column(column)
        rg = self._check_rg(row_group)
        err = _ffi.CarquetErrorStruct()
        bf = _ffi.get_lib().carquet_reader_get_bloom_filter(handle, rg, col.index, byref(err))
        return BloomFilter(bf, col) if bf else None

    def column_index(self, column: Union[int, str], row_group: int = 0) -> Optional[ColumnIndex]:
        """Per-page statistics (Parquet ColumnIndex) of a column chunk, or ``None``."""
        return ColumnIndex._read(self._require_open(), self._check_rg(row_group), self._column(column))

    def offset_index(self, column: Union[int, str], row_group: int = 0) -> Optional[OffsetIndex]:
        """Per-page file locations (Parquet OffsetIndex) of a column chunk, or ``None``."""
        return OffsetIndex._read(self._require_open(), self._check_rg(row_group), self._column(column))

    # -- projection --------------------------------------------------------
    def _resolve(self, columns: Optional[Sequence[str]]) -> list:
        """Top-level field ordinals for ``columns`` (all fields when ``None``)."""
        if columns is None:
            return [f.ordinal for f in self._fields]
        if isinstance(columns, str):
            raise TypeError("columns must be a list of names, not a str")
        out: list = []
        for name in columns:
            if not isinstance(name, str):
                raise TypeError(f"column names must be str, got {name!r}")
            f = self._field_by_name.get(name)
            if f is None:
                try:
                    top = self.parquet_schema.column(name).parts[0]
                except KeyError:
                    top = name.split(".", 1)[0] if self.parquet_schema.leaves_of(name) else None
                f = self._field_by_name.get(top) if top else None
            if f is None:
                raise KeyError(f"column {name!r} not found in {self._name} (fields: {[x.name for x in self._fields]})")
            if f.ordinal not in out:
                out.append(f.ordinal)
        return out

    def _converter(self, ordinals: list) -> Converter:
        key = tuple(ordinals)
        conv = self._converters.get(key)
        if conv is None:
            pa = _pa()
            raw = self._raw_schema()
            raw_proj = pa.schema([raw.field(i) for i in ordinals])
            conv = Converter(
                pa,
                raw_proj,
                [self._fields[i].leaves for i in ordinals],
                self.key_value_metadata,
                restore_arrow_schema=self._restore,
            )
            conv.raw_schema = raw_proj
            self._converters[key] = conv
        return conv

    def _row_groups(self, row_groups: Optional[Iterable[int]], filters: Optional[Filters]) -> list:
        if row_groups is None:
            rgs = list(range(self.num_row_groups))
        else:
            rgs = [self._check_rg(int(i)) for i in row_groups]
        if filters is not None and filters.dnf:
            pruner = RowGroupPruner(self._require_open(), self.parquet_schema, filters)
            rgs = [rg for rg in rgs if pruner.might_match(rg)]
        return rgs

    def filter_row_groups(self, filters: Any) -> list:
        """Indices of the row groups that may contain rows matching ``filters``.

        Uses only column statistics (no data is read); a returned row group may
        still hold no matching row.
        """
        return self._row_groups(None, normalize_filters(filters))

    # -- raw streams ---------------------------------------------------------
    def _stream(self, ordinals: list, rgs: list, batch_size: int, *, page_filter=None,
                row_range=None, use_threads: bool = True) -> Iterator:
        """Raw (unconverted) RecordBatches of the fields ``ordinals``."""
        fields = [self._fields[i] for i in ordinals]
        if not fields:
            return self._row_count_stream(rgs, batch_size)
        if all(f.batchable for f in fields):
            return _BatchStream(self, fields, rgs, batch_size, page_filter, row_range, use_threads)
        return self._nested_stream(ordinals, rgs, batch_size)

    def _row_count_stream(self, rgs: list, batch_size: Optional[int]) -> Iterator:
        """Zero-column batches carrying only row counts (``columns=[]``)."""
        pa = _pa()
        md = self.metadata
        step = batch_size or _READ_BATCH_SIZE
        for rg in rgs:
            n = md.row_group(rg).num_rows
            for start in range(0, n, step):
                yield pa.record_batch([pa.nulls(min(step, n - start))], names=["_"]).select([])

    def _nested_stream(self, ordinals: list, rgs: list, batch_size: Optional[int]) -> Iterator:
        pa = _pa()
        raw = self._converter(ordinals).raw_schema
        for rg in rgs:
            batch = self._read_arrow_fields(pa, rg, ordinals, raw)
            if batch_size is None or batch.num_rows <= batch_size:
                yield batch
            else:
                for start in range(0, batch.num_rows, batch_size):
                    yield batch.slice(start, batch_size)

    def _read_arrow_fields(self, pa, rg: int, ordinals: list, raw_schema) -> "pa.RecordBatch":
        lib = _ffi.get_lib()
        with self._lock:
            handle = self._require_open()
            array = ArrowArray()
            err = _ffi.CarquetErrorStruct()
            if _ffi.has_symbol("carquet_reader_read_arrow_fields"):
                idx = (c_int32 * len(ordinals))(*ordinals)
                st = lib.carquet_reader_read_arrow_fields(handle, rg, idx, len(ordinals), None, byref(array), byref(err))
                _ffi.check(st, err, self._name)
                return _import_batch(pa, array, raw_schema)
            # library without field projection: read everything, then select
            st = lib.carquet_reader_read_arrow(handle, rg, None, byref(array), byref(err))
            _ffi.check(st, err, self._name)
            full = _import_batch(pa, array, self._raw_schema())
            return pa.RecordBatch.from_arrays([full.column(i) for i in ordinals], schema=raw_schema)

    # -- reading -------------------------------------------------------------
    def read(
        self,
        columns: Optional[Sequence[str]] = None,
        *,
        filters: Any = None,
        row_groups: Optional[Iterable[int]] = None,
        use_threads: bool = True,
    ) -> "pa.Table":
        """Read the file (or some row groups) into a :class:`pyarrow.Table`.

        Args:
            columns: top-level fields to read, in the order wanted (a dotted
                path selects its top-level field). ``None`` reads every field.
            filters: row filter in DNF form, e.g. ``[("year", ">=", 2020),
                ("country", "in", ["FR", "DE"])]`` (see :mod:`carquet.filters`).
                Row groups and pages ruled out by statistics are not read; the
                result is filtered exactly.
            row_groups: only read these row groups (in this order).
            use_threads: allow multi-threaded decoding.
        """
        pa = _pa()
        flt = normalize_filters(filters)
        ordinals = self._resolve(columns)
        read_ordinals = self._with_filter_columns(ordinals, flt)
        rgs = self._row_groups(row_groups, flt)
        conv = self._converter(read_ordinals)
        page_filter = self._page_filter(read_ordinals, rgs, flt)
        batches = list(self._stream(read_ordinals, rgs, _READ_BATCH_SIZE, page_filter=page_filter,
                                    use_threads=use_threads))
        table = conv.table(pa.Table.from_batches(batches, schema=conv.raw_schema))
        return self._finish(table, ordinals, read_ordinals, flt)

    def read_row_group(self, i: int, columns: Optional[Sequence[str]] = None, *, use_threads: bool = True) -> "pa.Table":
        """Read row group ``i`` into a :class:`pyarrow.Table`."""
        return self.read(columns, row_groups=[i], use_threads=use_threads)

    def read_row_groups(self, row_groups: Iterable[int], columns: Optional[Sequence[str]] = None, *,
                        use_threads: bool = True) -> "pa.Table":
        """Read the given row groups (in that order) into one :class:`pyarrow.Table`."""
        return self.read(columns, row_groups=list(row_groups), use_threads=use_threads)

    def read_all(self, columns: Optional[Sequence[str]] = None, **kwargs) -> "pa.Table":
        """Alias of :meth:`read`."""
        return self.read(columns, **kwargs)

    def iter_batches(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        columns: Optional[Sequence[str]] = None,
        *,
        filters: Any = None,
        row_groups: Optional[Iterable[int]] = None,
        use_threads: bool = True,
    ) -> Iterator["pa.RecordBatch"]:
        """Stream the file as :class:`pyarrow.RecordBatch` es of at most ``batch_size`` rows.

        Batches never span row groups. With ``filters``, each batch is filtered
        exactly (so batches may be shorter, and empty ones are skipped).
        """
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        pa = _pa()
        flt = normalize_filters(filters)
        ordinals = self._resolve(columns)
        read_ordinals = self._with_filter_columns(ordinals, flt)
        rgs = self._row_groups(row_groups, flt)
        conv = self._converter(read_ordinals)
        page_filter = self._page_filter(read_ordinals, rgs, flt)
        expr = flt.to_expression(self.parquet_schema.fields) if flt is not None else None
        out_schema = None
        for raw in self._stream(read_ordinals, rgs, int(batch_size), page_filter=page_filter, use_threads=use_threads):
            batch = conv.batch(raw)
            if expr is not None:
                batch = batch.filter(expr)
                if batch.num_rows == 0:
                    continue
            if read_ordinals != ordinals:
                if out_schema is None:
                    out_schema = pa.schema([batch.schema.field(i) for i in range(len(ordinals))],
                                           metadata=batch.schema.metadata)
                batch = pa.RecordBatch.from_arrays(batch.columns[: len(ordinals)], schema=out_schema)
            yield batch

    def iter_row_groups(self, columns: Optional[Sequence[str]] = None, **kwargs) -> Iterator["pa.Table"]:
        """Yield one :class:`pyarrow.Table` per row group."""
        for i in range(self.num_row_groups):
            yield self.read(columns, row_groups=[i], **kwargs)

    def slice(self, offset: int, length: Optional[int] = None, columns: Optional[Sequence[str]] = None, *,
              use_threads: bool = True) -> "pa.Table":
        """Read rows ``[offset, offset + length)`` (to the end when ``length`` is None).

        Row groups outside the window are not read; for flat columns, pages
        before ``offset`` are skipped through the offset index when present.
        """
        pa = _pa()
        offset = int(offset)
        if offset < 0:
            offset = max(0, self.num_rows + offset)
        if length is not None and int(length) < 0:
            raise ValueError("length must be >= 0")
        end = self.num_rows if length is None else min(self.num_rows, offset + int(length))
        ordinals = self._resolve(columns)
        conv = self._converter(ordinals)
        if offset >= end:
            return conv.schema.empty_table()
        fields = [self._fields[i] for i in ordinals]
        if fields and all(f.batchable and f.leaves[0].max_repetition_level == 0 for f in fields):
            stream = _BatchStream(self, fields, list(range(self.num_row_groups)), _READ_BATCH_SIZE, None,
                                  (offset, end - offset), use_threads)
            table = pa.Table.from_batches(list(stream), schema=conv.raw_schema)
        else:
            rgs, first_row, row = [], None, 0
            for rg in range(self.num_row_groups):
                n = self.metadata.row_group(rg).num_rows
                if row + n > offset and row < end:
                    rgs.append(rg)
                    if first_row is None:
                        first_row = row
                row += n
            batches = list(self._stream(ordinals, rgs, _READ_BATCH_SIZE, use_threads=use_threads))
            table = pa.Table.from_batches(batches, schema=conv.raw_schema)
            table = table.slice(offset - (first_row or 0), end - offset)
        return conv.table(table)

    def head(self, n: int = 5, columns: Optional[Sequence[str]] = None) -> "pa.Table":
        """The first ``n`` rows."""
        return self.slice(0, n, columns)

    def count_rows(self, filters: Any = None) -> int:
        """Number of rows, or of rows matching ``filters`` (only the filter columns are read)."""
        flt = normalize_filters(filters)
        if flt is None:
            return self.num_rows
        cols = flt.columns if flt.dnf else None
        return sum(b.num_rows for b in self.iter_batches(_READ_BATCH_SIZE, columns=cols, filters=filters))

    # -- helpers -------------------------------------------------------------
    def _with_filter_columns(self, ordinals: list, flt: Optional[Filters]) -> list:
        if flt is None or not flt.dnf:
            if flt is not None and flt.expression is not None and len(ordinals) != len(self._fields):
                # an Expression may reference any field: read them all, drop afterwards
                return ordinals + [f.ordinal for f in self._fields if f.ordinal not in ordinals]
            return ordinals
        extra = []
        for name in flt.columns:
            for o in self._resolve([name]):
                if o not in ordinals and o not in extra:
                    extra.append(o)
        return ordinals + extra

    def _page_filter(self, ordinals: list, rgs: list, flt: Optional[Filters]):
        if flt is None or not flt.dnf or len(flt.dnf) != 1:
            return None
        if not all(self._fields[i].batchable for i in ordinals):
            return None
        pf = build_page_filter(self.parquet_schema, flt)
        if pf is None:
            return None
        md = self.metadata
        for rg in rgs:
            for c in pf.columns:
                if not md.row_group(rg).column(c).has_offset_index:
                    return None
        return pf

    def _finish(self, table, ordinals: list, read_ordinals: list, flt: Optional[Filters]):
        if flt is not None:
            table = table.filter(flt.to_expression(self.parquet_schema.fields))
        if read_ordinals != ordinals:
            table = table.select(list(range(len(ordinals))))
        return table


class _BatchStream:
    """Iterator over raw RecordBatches from a C batch reader."""

    def __init__(self, pf: ParquetFile, fields: list, rgs: list, batch_size: int, page_filter, row_range,
                 use_threads: bool) -> None:
        self._pf = pf
        self._pa = _pa()
        self._raw_schema = pf._converter([f.ordinal for f in fields]).raw_schema
        self._leaves = [c.index for f in fields for c in f.leaves]
        self._batch_size = max(1, min(int(batch_size), 2**31 - 1))
        self._page_filter = page_filter
        self._row_range = row_range
        self._use_threads = use_threads
        self._br: Optional[int] = None
        self._keep: Any = None
        self._done = False
        self._interrupted = False
        # One C reader over all selected row groups when they are in file order;
        # otherwise one reader per row group, in the order asked.
        rgs = list(rgs)
        if not rgs:
            self._runs = []
        elif row_range is not None or rgs == list(range(pf.num_row_groups)):
            self._runs = [None]  # every row group
        elif rgs == sorted(set(rgs)):
            self._runs = [rgs]
        else:
            self._runs = [[rg] for rg in rgs]
        pf._streams.add(self)

    def __iter__(self) -> "_BatchStream":
        return self

    def _open(self, run: Optional[list]) -> None:
        lib = _ffi.get_lib()
        handle = self._pf._require_open()
        cfg = _ffi.BatchReaderConfig()
        lib.carquet_batch_reader_config_init(byref(cfg))
        cfg.batch_size = self._batch_size
        threads = self._pf._num_threads
        cfg.num_threads = threads if self._use_threads else 1
        cfg.use_mmap = self._pf._memory_map
        idx = (c_int32 * len(self._leaves))(*self._leaves)
        cfg.column_indices = idx
        cfg.num_columns = len(self._leaves)
        keep: list = [idx]
        if run is not None:
            wanted = frozenset(run)
            cb = _ffi.RowGroupFilterFn(lambda _r, i, _u: i in wanted)
            cfg.row_group_filter = cb
            keep.append(cb)
        err = _ffi.CarquetErrorStruct()
        br = lib.carquet_batch_reader_create(handle, byref(cfg), byref(err))
        if not br:
            _ffi.check(err.code or StatusCode.INTERNAL, err, self._pf._name)
        try:
            if self._page_filter is not None:
                _ffi.check(lib.carquet_batch_reader_set_page_filter(br, self._page_filter.clauses,
                                                                    self._page_filter.count), None, "page filter")
            if self._row_range is not None:
                off, lim = self._row_range
                _ffi.check(lib.carquet_batch_reader_set_row_range(br, int(off), int(lim)), None, "row range")
        except BaseException:
            lib.carquet_batch_reader_free(br)
            raise
        self._br = br
        self._keep = keep

    def __next__(self) -> "pa.RecordBatch":
        lib = _ffi.get_lib()
        pa = self._pa
        with self._pf._lock:
            while True:
                if self._done:
                    if self._interrupted:
                        raise CarquetError(StatusCode.ALREADY_CLOSED,
                                           f"{self._pf._name}: file closed while batches were being read")
                    raise StopIteration
                if self._br is None:
                    if not self._runs:
                        self._done = True
                        self.close()
                        raise StopIteration
                    self._open(self._runs.pop(0))
                batch = c_void_p()
                st = lib.carquet_batch_reader_next(self._br, byref(batch))
                if st == _ffi.END_OF_DATA or (st == _ffi.OK and not batch):
                    self._free_reader()
                    continue
                if st != _ffi.OK:
                    detail = lib.carquet_batch_reader_last_error(self._br)
                    exc = _ffi.make_error(st, detail.contents if detail else None, self._pf._name)
                    self.close()
                    raise exc
                array = ArrowArray()
                err = _ffi.CarquetErrorStruct()
                st = lib.carquet_arrow_export_batch(batch, self._pf._schema_handle, None, byref(array), byref(err))
                lib.carquet_row_batch_free(batch)
                if st != _ffi.OK:
                    self.close()
                    _ffi.check(st, err, self._pf._name)
                return _import_batch(pa, array, self._raw_schema)

    def _free_reader(self) -> None:
        br, self._br = self._br, None
        if br:
            _ffi.get_lib().carquet_batch_reader_free(br)
        self._keep = None

    def close(self, interrupted: bool = False) -> None:
        if interrupted and not self._done:
            self._interrupted = True
        self._done = True
        self._runs = []
        self._free_reader()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Module-level functions
# --------------------------------------------------------------------------
def _sources(source: Any) -> Optional[list]:
    """Expand a directory or a list of paths into files; ``None`` for a single source."""
    if isinstance(source, (list, tuple)):
        return list(source)
    if _is_path(source) and os.path.isdir(source):
        out = []
        for root, dirs, files in os.walk(source):
            dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
            for name in sorted(files):
                if not name.startswith((".", "_")) and name.endswith((".parquet", ".parq", ".pq")):
                    out.append(os.path.join(root, name))
        return out
    return None


def read_table(
    source: Source,
    *,
    columns: Optional[Sequence[str]] = None,
    filters: Any = None,
    row_groups: Optional[Iterable[int]] = None,
    use_threads: bool = True,
    memory_map: bool = True,
    verify_checksums: bool = True,
    restore_arrow_schema: bool = True,
    use_mmap: Optional[bool] = None,
) -> "pa.Table":
    """Read a Parquet file into a :class:`pyarrow.Table`.

    ``source`` is a path, a bytes-like object, a binary file object, a
    directory (every ``*.parquet`` file under it, in sorted order) or a list
    of any of these; several files are concatenated (their schemas must
    match). See :meth:`ParquetFile.read` for ``columns``, ``filters`` and
    ``row_groups``.
    """
    many = _sources(source)
    opts = dict(memory_map=memory_map, verify_checksums=verify_checksums,
                restore_arrow_schema=restore_arrow_schema, use_mmap=use_mmap)
    if many is not None:
        if row_groups is not None:
            raise ValueError("row_groups cannot be combined with several source files")
        pa = _pa()
        if not many:
            raise ValueError(f"no Parquet files found in {source!r}")
        tables = [read_table(s, columns=columns, filters=filters, use_threads=use_threads, **opts) for s in many]
        return pa.concat_tables(tables)
    with ParquetFile(source, **opts) as pf:
        return pf.read(columns, filters=filters, row_groups=row_groups, use_threads=use_threads)


def read_pandas(source: Source, **kwargs):
    """:func:`read_table` followed by ``Table.to_pandas()`` (pandas metadata restores the index)."""
    return read_table(source, **kwargs).to_pandas()


def iter_batches(
    source: Source,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    columns: Optional[Sequence[str]] = None,
    filters: Any = None,
    row_groups: Optional[Iterable[int]] = None,
    use_threads: bool = True,
    memory_map: bool = True,
    verify_checksums: bool = True,
) -> Iterator["pa.RecordBatch"]:
    """Stream a Parquet file (or several) as RecordBatches; the file is closed at the end."""
    many = _sources(source)
    for s in many if many is not None else [source]:
        with ParquetFile(s, memory_map=memory_map, verify_checksums=verify_checksums) as pf:
            yield from pf.iter_batches(batch_size, columns, filters=filters, row_groups=row_groups,
                                       use_threads=use_threads)


def read_schema(source: Source) -> "pa.Schema":
    """The Arrow schema of a Parquet file, without reading any data."""
    with ParquetFile(source, memory_map=False) as pf:
        return pf.schema


def read_metadata(source: Source) -> FileMetaData:
    """The footer metadata (:class:`FileMetaData`) of a Parquet file, without reading data."""
    with ParquetFile(source, memory_map=False) as pf:
        return pf.metadata


def read_parquet_schema(source: Source) -> ParquetSchema:
    """The Parquet-level schema (:class:`ParquetSchema`) of a file; works without PyArrow."""
    with ParquetFile(source, memory_map=False) as pf:
        return pf.parquet_schema

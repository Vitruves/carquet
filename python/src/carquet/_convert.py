"""Post-read conversions that make carquet's Arrow output match ``pyarrow.parquet``.

The C bridge exports a few Parquet types by their physical storage (DECIMAL as
fixed-size binary / int32 / int64, INT96 as 12-byte binary). This module turns
them into ``decimal128`` / ``decimal256`` and ``timestamp[ns]`` with vectorised
NumPy byte shuffles (no per-value Python work), then restores the original
Arrow types recorded in the ``ARROW:schema`` footer entry (time zones, large
types, dictionaries, field names of list children, field metadata, pandas
metadata, ...).

A :class:`Converter` is planned once per (file, projection) and applied to
every batch, so the per-batch cost is zero when nothing needs converting.
"""

from __future__ import annotations

import base64
import decimal
from typing import Any, Optional

from .schema import ColumnSchema

ARROW_SCHEMA_KEY = "ARROW:schema"
_NS_PER_DAY = 86400 * 10**9
_JULIAN_EPOCH_DAY = 2440588


def stored_arrow_schema(pa, kv: dict) -> Optional[Any]:
    """Decode the ``ARROW:schema`` footer entry (base64 IPC schema), if any."""
    raw = kv.get(ARROW_SCHEMA_KEY)
    if not raw:
        return None
    try:
        blob = base64.b64decode(raw)
        return pa.ipc.read_schema(pa.py_buffer(blob))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Leaf conversions
# --------------------------------------------------------------------------
def _decimal_type(pa, precision: int, scale: int):
    if precision <= 38:
        return pa.decimal128(precision, scale)
    return pa.decimal256(precision, scale)


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def _leaf_plan(pa, arrow_type, col: ColumnSchema):
    """``(new_type, fn)`` for a leaf that needs converting, else ``None``.

    Without NumPy the column keeps its storage type (fixed-size binary / int /
    12-byte binary)."""
    lt = col.logical_type
    if (lt is not None and lt.name == "DECIMAL") or col.physical_type == "INT96":
        if not _has_numpy():
            return None
    if lt is not None and lt.name == "DECIMAL" and lt.precision:
        target = _decimal_type(pa, lt.precision, lt.scale or 0)
        if pa.types.is_fixed_size_binary(arrow_type):
            width = arrow_type.byte_width
            if width > target.byte_width:
                return None
            return target, lambda arr: _flba_to_decimal(pa, arr, width, target)
        if pa.types.is_int32(arrow_type) or pa.types.is_int64(arrow_type):
            return target, lambda arr: _int_to_decimal(pa, arr, target)
        if pa.types.is_binary(arrow_type):
            return target, lambda arr: _bytes_to_decimal(pa, arr, target, lt.scale or 0)
        return None
    if col.physical_type == "INT96" and pa.types.is_fixed_size_binary(arrow_type) and arrow_type.byte_width == 12:
        target = pa.timestamp("ns")
        return target, lambda arr: _int96_to_timestamp(pa, arr, target)
    return None


def _np():
    import numpy as np

    return np


def _flba_to_decimal(pa, arr, width: int, target):
    np = _np()
    total = arr.offset + len(arr)
    out_w = target.byte_width
    validity, data = arr.buffers()[:2]
    src = np.frombuffer(data, dtype=np.uint8, count=total * width).reshape(total, width)
    out = np.empty((total, out_w), dtype=np.uint8)
    if width:
        negative = (src[:, 0] & 0x80) != 0
        out[:, width:] = np.where(negative, 0xFF, 0x00).astype(np.uint8)[:, None]
        out[:, :width] = src[:, ::-1]  # big-endian two's complement -> little-endian
    else:
        out[:] = 0
    return pa.Array.from_buffers(target, len(arr), [validity, pa.py_buffer(out)], arr.null_count, arr.offset)


def _int_to_decimal(pa, arr, target):
    np = _np()
    total = arr.offset + len(arr)
    dtype = np.int32 if pa.types.is_int32(arr.type) else np.int64
    validity, data = arr.buffers()[:2]
    vals = np.frombuffer(data, dtype=dtype, count=total).astype(np.int64)
    limbs = target.byte_width // 8
    out = np.empty((total, limbs), dtype=np.int64)
    out[:, 0] = vals
    out[:, 1:] = (vals >> 63)[:, None]
    return pa.Array.from_buffers(target, len(arr), [validity, pa.py_buffer(out)], arr.null_count, arr.offset)


def _bytes_to_decimal(pa, arr, target, scale: int):
    vals = [
        None if v is None else decimal.Decimal(int.from_bytes(v, "big", signed=True)).scaleb(-scale)
        for v in arr.to_pylist()
    ]
    return pa.array(vals, type=target)


def _int96_to_timestamp(pa, arr, target):
    np = _np()
    total = arr.offset + len(arr)
    validity, data = arr.buffers()[:2]
    raw = np.frombuffer(data, dtype=np.uint8, count=total * 12).reshape(total, 12)
    nanos = np.ascontiguousarray(raw[:, :8]).view("<i8").ravel()
    days = np.ascontiguousarray(raw[:, 8:]).view("<u4").ravel().astype(np.int64)
    ns = (days - _JULIAN_EPOCH_DAY) * _NS_PER_DAY + nanos
    return pa.Array.from_buffers(target, len(arr), [validity, pa.py_buffer(ns)], arr.null_count, arr.offset)


# --------------------------------------------------------------------------
# Nested plans: mirror the Arrow type tree; the leaves are visited in the same
# depth-first order as the Parquet leaf columns of the field.
# --------------------------------------------------------------------------
def _plan(pa, arrow_type, leaves: list):
    """Return ``(new_type, fn_or_None)``; consumes this subtree's leaves from ``leaves``."""
    t = arrow_type
    if pa.types.is_struct(t):
        kids = [_plan(pa, t.field(i).type, leaves) for i in range(t.num_fields)]
        if all(fn is None for _, fn in kids):
            return t, None
        new_t = pa.struct([t.field(i).with_type(kids[i][0]) for i in range(t.num_fields)])

        def conv_struct(arr, kids=kids, new_t=new_t):
            # StructArray.field() applies the parent's offset: rebuild at offset 0
            children = [fn(arr.field(i)) if fn else arr.field(i) for i, (_, fn) in enumerate(kids)]
            mask = arr.is_null() if arr.null_count else None
            return pa.StructArray.from_arrays(children, fields=list(new_t), mask=mask)

        return new_t, conv_struct
    if pa.types.is_map(t):
        key_t, key_fn = _plan(pa, t.key_type, leaves)
        item_t, item_fn = _plan(pa, t.item_type, leaves)
        if key_fn is None and item_fn is None:
            return t, None
        new_t = pa.map_(t.key_field.with_type(key_t), t.item_field.with_type(item_t), keys_sorted=t.keys_sorted)
        entries_t = pa.struct([new_t.key_field, new_t.item_field])

        def conv_map(arr, new_t=new_t, entries_t=entries_t):
            entries = arr.values if hasattr(arr, "values") else arr.flatten()
            keys = entries.field(0)
            items = entries.field(1)
            keys = key_fn(keys) if key_fn else keys
            items = item_fn(items) if item_fn else items
            new_entries = pa.StructArray.from_arrays([keys, items], fields=list(entries_t))
            return pa.Array.from_buffers(new_t, len(arr), arr.buffers()[:2], arr.null_count, arr.offset, [new_entries])

        return new_t, conv_map
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        val_t, val_fn = _plan(pa, t.value_type, leaves)
        if val_fn is None:
            return t, None
        field = t.value_field.with_type(val_t)
        new_t = pa.list_(field) if pa.types.is_list(t) else pa.large_list(field)

        def conv_list(arr, new_t=new_t):
            return pa.Array.from_buffers(
                new_t, len(arr), arr.buffers()[:2], arr.null_count, arr.offset, [val_fn(arr.values)]
            )

        return new_t, conv_list
    if pa.types.is_fixed_size_list(t):
        val_t, val_fn = _plan(pa, t.value_type, leaves)
        if val_fn is None:
            return t, None
        new_t = pa.list_(t.value_field.with_type(val_t), t.list_size)

        def conv_fsl(arr, new_t=new_t):
            return pa.Array.from_buffers(
                new_t, len(arr), arr.buffers()[:1], arr.null_count, arr.offset, [val_fn(arr.values)]
            )

        return new_t, conv_fsl
    # primitive leaf
    if not leaves:
        return t, None
    col = leaves.pop(0)
    plan = _leaf_plan(pa, t, col)
    if plan is None:
        return t, None
    return plan


class Converter:
    """Batch/table post-processor for one projection of one file.

    ``raw_schema`` is what the C bridge produces; :attr:`schema` is the final
    schema callers see. ``leaves_per_field[i]`` lists the Parquet leaf columns
    of output field ``i`` in order.
    """

    def __init__(self, pa, raw_schema, leaves_per_field: list, kv: dict, *, restore_arrow_schema: bool = True) -> None:
        self._pa = pa
        stored = stored_arrow_schema(pa, kv) if restore_arrow_schema else None
        fields = []
        self._steps: list = []  # (column position, fn or None, cast target or None)
        for i, f in enumerate(raw_schema):
            new_t, fn = _plan(pa, f.type, list(leaves_per_field[i]))
            target = None
            new_f = f.with_type(new_t)
            if stored is not None:
                idx = stored.get_field_index(f.name)
                if idx >= 0:
                    sf = stored.field(idx)
                    if sf.type != new_t and (_castable(pa, new_t, sf.type) or pa.types.is_null(sf.type)):
                        target = sf.type
                    new_f = pa.field(f.name, target if target is not None else new_t, nullable=f.nullable,
                                     metadata=sf.metadata)
            if fn is not None or target is not None:
                self._steps.append((i, fn, target))
            fields.append(new_f)
        metadata = {k: v for k, v in kv.items() if k != ARROW_SCHEMA_KEY and v is not None}
        if stored is not None and stored.metadata:
            for k, v in stored.metadata.items():
                metadata.setdefault(k.decode("utf-8", "replace"), v.decode("utf-8", "replace"))
        self.schema = pa.schema(fields, metadata=metadata or None)
        self._identity = not self._steps

    def _convert_array(self, arr, fn, target):
        if fn is not None:
            arr = fn(arr)
        if target is not None:
            arr = _cast(self._pa, arr, target)
        return arr

    def _schema_for(self, arrays):
        """The planned schema, or -- if a cast had to be skipped at runtime --
        the same schema with the types actually produced."""
        if all(a.type == f.type for a, f in zip(arrays, self.schema)):
            return self.schema
        pa = self._pa
        return pa.schema([f.with_type(a.type) for a, f in zip(arrays, self.schema)], metadata=self.schema.metadata)

    def batch(self, batch):
        pa = self._pa
        if not len(self.schema):  # columns=[]: keep the row count
            return batch.replace_schema_metadata(self.schema.metadata)
        if self._identity:
            return pa.RecordBatch.from_arrays(batch.columns, schema=self.schema)
        cols = list(batch.columns)
        for i, fn, target in self._steps:
            cols[i] = self._convert_array(cols[i], fn, target)
        return pa.RecordBatch.from_arrays(cols, schema=self._schema_for(cols))

    def table(self, table):
        pa = self._pa
        if not len(self.schema):  # (replace_schema_metadata would drop the row count here)
            return pa.table({"_": pa.nulls(table.num_rows)}).replace_schema_metadata(self.schema.metadata).select([])
        cols = list(table.columns)
        for i, fn, target in self._steps:
            chunks = [self._convert_array(c, fn, target) for c in cols[i].chunks]
            types = {c.type for c in chunks}
            if len(types) > 1:  # a cast failed for some chunks only: keep every chunk unconverted
                chunks = [self._convert_array(c, fn, None) for c in cols[i].chunks]
            cols[i] = pa.chunked_array(chunks, type=chunks[0].type if chunks else self.schema.field(i).type)
        return pa.Table.from_arrays(cols, schema=self._schema_for(cols))


def _cast(pa, arr, target):
    """Cast to a type restored from ARROW:schema; keep ``arr`` when the data does not allow it."""
    if pa.types.is_null(target):
        return pa.nulls(len(arr)) if arr.null_count == len(arr) else arr
    if pa.types.is_fixed_size_list(target) and (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)) \
            and arr.null_count:
        return _list_to_fixed_size(pa, arr, target)
    for safe in (True, False):
        try:
            return arr.cast(target, safe=safe)
        except Exception:
            continue
    return arr


def _list_to_fixed_size(pa, arr, target):
    """list -> fixed_size_list when some lists are null (stored without child values)."""
    import pyarrow.compute as pc

    try:
        filler = pa.scalar([None] * target.list_size, type=arr.type)
        dense = pc.fill_null(arr, filler).cast(target)
        validity = pc.invert(arr.is_null())
        return pa.FixedSizeListArray.from_arrays(dense.values, type=target, mask=pc.invert(validity))
    except Exception:
        return arr


def _castable(pa, src, dst) -> bool:
    try:
        pa.array([], type=src).cast(dst)
        return True
    except Exception:
        return False

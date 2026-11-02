"""Row filters: normalisation, predicate pushdown and exact evaluation.

``filters`` follows the ``pyarrow.parquet`` DNF convention:

* a list of ``(column, op, value)`` tuples is a conjunction (AND);
* a list of such lists is a disjunction (OR) of conjunctions;
* a ``pyarrow.compute.Expression`` is also accepted (evaluated exactly, without
  pushdown).

Supported ops: ``==`` (``=``), ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in``,
``not in``, ``is null``, ``is not null`` (the last two take no value; a
``("x", "==", None)`` / ``("x", "!=", None)`` clause means the same).

Filters are applied in three stages, so the result is always exact:

1. row groups whose column statistics rule the predicate out are skipped
   (``carquet_reader_row_group_matches``);
2. for a single conjunction on flat columns of a file written with a page
   index, pages whose min/max rule it out are skipped as well
   (``carquet_batch_reader_set_page_filter``);
3. the surviving rows are filtered exactly with ``pyarrow.compute``.
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_bool
from typing import Any, Optional, Sequence

from . import _ffi
from ._values import encode_value, is_float_column, is_nan
from .schema import ColumnSchema, ParquetSchema

__all__ = ["normalize_filters"]

_OPS = {
    "==": "==",
    "=": "==",
    "eq": "==",
    "!=": "!=",
    "<>": "!=",
    "ne": "!=",
    "<": "<",
    "lt": "<",
    "<=": "<=",
    "le": "<=",
    ">": ">",
    "gt": ">",
    ">=": ">=",
    "ge": ">=",
    "in": "in",
    "not in": "not in",
    "not_in": "not in",
    "is null": "is null",
    "is_null": "is null",
    "is not null": "is not null",
    "is_not_null": "is not null",
}

# carquet_compare_op_t
_COMPARE = {"==": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}
# carquet_filter_op_t
_FILTER = {"==": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5, "in": 7, "is null": 8, "is not null": 9}
_PACKED = ("BOOLEAN", "INT32", "INT64", "FLOAT", "DOUBLE", "FIXED_LEN_BYTE_ARRAY")
_NOT_PUSHABLE = object()


def _encode_clause(col: ColumnSchema, op: str, val):
    """Encoded value(s) for a pushable clause, or ``_NOT_PUSHABLE``.

    Pruning must never drop a row the exact filter would keep, so a clause is
    not pushed down when: its value is inexact in the column type (float32,
    coarser time unit, datetime vs date), out of range or of the wrong type;
    it is ``!=`` on a float column (statistics skip NaN, which ``!=`` keeps);
    or a value is NaN or None.
    """
    if op in ("is null", "is not null"):
        return None
    if op == "not in" or (op == "!=" and is_float_column(col)):
        return _NOT_PUSHABLE
    vals = val if op == "in" else [val]
    if any(v is None or is_nan(v) for v in vals):
        return _NOT_PUSHABLE
    try:
        enc = [encode_value(v, col, exact=True) for v in vals]
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        return _NOT_PUSHABLE
    return enc if op == "in" else enc[0]


class Filters:
    """A normalised filter: ``dnf`` (list of conjunctions) or an ``expression``."""

    def __init__(self, dnf: Optional[list] = None, expression: Any = None) -> None:
        self.dnf = dnf
        self.expression = expression

    @property
    def columns(self) -> list:
        """Column names the predicate references (DNF form only), in first-use order."""
        out: list = []
        for conj in self.dnf or ():
            for col, _op, _v in conj:
                if col not in out:
                    out.append(col)
        return out

    def to_expression(self, top_level: Sequence[str]):
        if self.expression is not None:
            return self.expression
        import pyarrow.compute as pc

        tops = set(top_level)

        def ref(name: str):
            if name in tops or "." not in name:
                return pc.field(name)
            return pc.field(*name.split("."))

        disj = None
        for conj in self.dnf:
            e = None
            for col, op, val in conj:
                f = ref(col)
                if op == "==":
                    c = f == val
                elif op == "!=":
                    c = f != val
                elif op == "<":
                    c = f < val
                elif op == "<=":
                    c = f <= val
                elif op == ">":
                    c = f > val
                elif op == ">=":
                    c = f >= val
                elif op == "in":
                    c = f.isin(list(val))
                elif op == "not in":
                    c = ~f.isin(list(val))
                elif op == "is null":
                    c = f.is_null()
                else:  # is not null
                    c = f.is_valid()
                e = c if e is None else (e & c)
            disj = e if disj is None else (disj | e)
        return disj


def normalize_filters(filters: Any) -> Optional[Filters]:
    if filters is None:
        return None
    if _is_expression(filters):
        return Filters(expression=filters)
    if isinstance(filters, tuple) and filters and isinstance(filters[0], str):
        filters = [filters]  # a single clause
    if not isinstance(filters, (list, tuple)):
        raise TypeError(
            "filters must be a list of (column, op, value) tuples, a list of such lists, "
            f"or a pyarrow.compute.Expression; got {type(filters).__name__}"
        )
    if not filters:
        return None
    if all(isinstance(c, tuple) for c in filters):
        groups = [list(filters)]
    elif all(isinstance(c, (list, tuple)) and (not c or isinstance(c[0], tuple)) for c in filters):
        groups = [list(c) for c in filters]
    else:
        raise TypeError("filters mixes clauses and clause lists; use [(...), ...] or [[(...)], [(...)]]")
    dnf = []
    for conj in groups:
        if not conj:
            raise ValueError("an empty conjunction in filters would match every row; remove it")
        dnf.append([_clause(c) for c in conj])
    return Filters(dnf=dnf)


def _is_expression(obj: Any) -> bool:
    t = type(obj)
    return t.__name__ == "Expression" and t.__module__.startswith("pyarrow")


def _clause(c: tuple) -> tuple:
    if len(c) == 2:
        col, op = c
        val = None
    elif len(c) == 3:
        col, op, val = c
    else:
        raise ValueError(f"filter clause must be (column, op, value) or (column, op); got {c!r}")
    if not isinstance(col, str):
        raise TypeError(f"filter column must be a str, got {col!r}")
    key = str(op).strip().lower()
    if key not in _OPS:
        raise ValueError(f"unsupported filter op {op!r}; expected one of {sorted(set(_OPS.values()))}")
    op = _OPS[key]
    if val is None and op == "==":
        op = "is null"
    elif val is None and op == "!=":
        op = "is not null"
    if op in ("in", "not in"):
        if isinstance(val, (str, bytes)) or not hasattr(val, "__iter__"):
            raise TypeError(f"'{op}' needs a collection of values, got {val!r}")
        val = list(val)
    elif op not in ("is null", "is not null") and len(c) == 2:
        raise ValueError(f"filter op {op!r} needs a value")
    return (col, op, val)


# --------------------------------------------------------------------------
# Pushdown
# --------------------------------------------------------------------------
def _leaf(schema: ParquetSchema, name: str) -> Optional[ColumnSchema]:
    try:
        col = schema.column(name)
    except KeyError:
        return None
    if col.max_repetition_level > 0 or col.physical_type == "INT96":
        return None
    return col


class RowGroupPruner:
    """Evaluates a DNF predicate against row-group statistics."""

    def __init__(self, handle: int, schema: ParquetSchema, filters: Filters) -> None:
        self._handle = handle
        self._schema = schema
        self._dnf = filters.dnf or []
        # (column, op, value) -> (leaf, [encoded values]) or None when not pushable
        self._prepared = []
        for conj in self._dnf:
            self._prepared.append([self._prepare(c) for c in conj])

    def _prepare(self, clause: tuple):
        col_name, op, val = clause
        col = _leaf(self._schema, col_name)
        if col is None:
            return None
        enc = _encode_clause(col, op, val)
        return None if enc is _NOT_PUSHABLE else (col, op, enc)

    def might_match(self, rg: int) -> bool:
        if not self._dnf:
            return True
        return any(all(self._clause_might_match(rg, p) for p in conj) for conj in self._prepared)

    def _matches(self, rg: int, col: ColumnSchema, op: str, raw: bytes) -> bool:
        lib = _ffi.get_lib()
        out = c_bool(True)
        buf = ctypes.create_string_buffer(raw, len(raw))
        st = lib.carquet_reader_row_group_matches(self._handle, rg, col.index, _COMPARE[op], buf, len(raw), byref(out))
        return True if st != _ffi.OK else bool(out.value)

    def _clause_might_match(self, rg: int, prepared) -> bool:
        if prepared is None:
            return True
        col, op, raw = prepared
        if op in ("is null", "is not null"):
            s = _ffi.ColumnStatisticsStruct()
            lib = _ffi.get_lib()
            if lib.carquet_reader_column_statistics(self._handle, rg, col.index, byref(s)) != _ffi.OK:
                return True
            if not s.has_null_count:
                return True
            if op == "is null":
                return int(s.null_count) > 0
            return int(s.null_count) < int(s.num_values) or int(s.num_values) <= 0
        if op == "in":
            return any(self._matches(rg, col, "==", r) for r in raw)
        return self._matches(rg, col, op, raw)


class PageFilter:
    """``carquet_filter_clause_t`` array for one conjunction (kept alive with its values)."""

    def __init__(self, clauses, keepalive: list, columns: list) -> None:
        self.clauses = clauses
        self.count = len(clauses)
        self._keepalive = keepalive
        self.columns = columns  # leaf ordinals referenced


def build_page_filter(schema: ParquetSchema, filters: Filters) -> Optional[PageFilter]:
    """Page-index clauses for a single conjunction, or ``None`` if nothing is pushable."""
    if not filters.dnf or len(filters.dnf) != 1:
        return None
    keep: list = []
    structs: list = []
    cols: list = []
    for col_name, op, val in filters.dnf[0]:
        col = _leaf(schema, col_name)
        if col is None or op not in _FILTER:
            continue
        enc = _encode_clause(col, op, val)
        if enc is _NOT_PUSHABLE or (op == "in" and not enc):
            continue
        c = _ffi.FilterClauseStruct()
        c.column_index = col.index
        c.op = _FILTER[op]
        if op == "in":
            if col.physical_type in _PACKED:  # native scalars / raw FLBA values, back to back
                blob = ctypes.create_string_buffer(b"".join(enc), sum(len(e) for e in enc))
                keep.append(blob)
                c.values = ctypes.addressof(blob)
            else:  # BYTE_ARRAY: carquet_byte_array_t entries
                arr = (_ffi.ByteArrayStruct * len(enc))()
                for i, e in enumerate(enc):
                    b = ctypes.create_string_buffer(e, len(e))
                    keep.append(b)
                    arr[i].data = ctypes.addressof(b)
                    arr[i].length = len(e)
                keep.append(arr)
                c.values = ctypes.addressof(arr)
            c.value_count = len(enc)
        elif enc is not None:
            b = ctypes.create_string_buffer(enc, len(enc))
            keep.append(b)
            c.value = ctypes.addressof(b)
            c.value_size = len(enc)
        structs.append(c)
        if col.index not in cols:
            cols.append(col.index)
    if not structs:
        return None
    arr = (_ffi.FilterClauseStruct * len(structs))(*structs)
    return PageFilter(arr, keep, cols)

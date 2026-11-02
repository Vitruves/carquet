"""carquet -- fast Apache Parquet reading and writing, backed by the carquet C library.

Reading::

    import carquet as cq

    table = cq.read_table("data.parquet", columns=["id", "price"],
                          filters=[("price", ">", 100)])
    df = cq.read_pandas("data.parquet")

    with cq.ParquetFile("data.parquet") as pf:
        pf.schema, pf.parquet_schema, pf.metadata
        for batch in pf.iter_batches(batch_size=50_000):
            ...

Writing::

    cq.write_table(table, "out.parquet", compression="zstd", use_dictionary=["country"])

    with cq.ParquetWriter("out.parquet", schema) as w:
        w.write(batch)

Tables cross the C boundary through the Arrow C Data Interface, and every C
call runs without the GIL. :func:`info`, :func:`validate`,
:func:`read_metadata`, :func:`read_parquet_schema`, :func:`cpu_info` and the
version helpers work without PyArrow.
"""

from __future__ import annotations

from . import _ffi
from ._types import COMPRESSION_CODECS
from .errors import CarquetError, StatusCode
from .metadata import (
    BloomFilter,
    ColumnChunkMetaData,
    ColumnIndex,
    FileMetaData,
    GeospatialStatistics,
    OffsetIndex,
    PageLocation,
    PageStatistics,
    RowGroupMetaData,
    Statistics,
)
from .reader import (
    FileInfo,
    ParquetFile,
    info,
    iter_batches,
    read_metadata,
    read_pandas,
    read_parquet_schema,
    read_schema,
    read_table,
    validate,
)
from .schema import ColumnSchema, LogicalType, ParquetSchema, SchemaElement
from .writer import ParquetWriter, write_table

__all__ = [
    "COMPRESSION_CODECS",
    "BloomFilter",
    "CarquetError",
    "ColumnChunkMetaData",
    "ColumnIndex",
    "ColumnSchema",
    "FileInfo",
    "FileMetaData",
    "GeospatialStatistics",
    "LogicalType",
    "OffsetIndex",
    "PageLocation",
    "PageStatistics",
    "ParquetFile",
    "ParquetSchema",
    "ParquetWriter",
    "Reader",
    "RowGroupMetaData",
    "SchemaElement",
    "Statistics",
    "StatusCode",
    "__version__",
    "cpu_info",
    "info",
    "iter_batches",
    "lib_version",
    "open_reader",
    "read_metadata",
    "read_pandas",
    "read_parquet_schema",
    "read_schema",
    "read_table",
    "schema",
    "validate",
    "write_table",
]


def lib_version() -> str:
    """Version string reported by the loaded C library (``carquet_version()``)."""
    return _ffi.get_lib().carquet_version().decode("ascii")


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("carquet")
    except Exception:  # not installed as a distribution (e.g. run from a checkout)
        try:
            return lib_version()
        except Exception:
            return "0.0.0"


__version__ = _package_version()


def cpu_info() -> dict:
    """SIMD features detected at runtime (which optimised kernels the library uses)."""
    c = _ffi.get_lib().carquet_get_cpu_info().contents
    return {name: (int(getattr(c, name)) if name == "sve_vector_length" else bool(getattr(c, name)))
            for name, _ in c._fields_}


# Names from carquet 0.7.2 and earlier, kept as aliases.
Reader = ParquetFile
schema = read_schema


def open_reader(path, **kwargs) -> ParquetFile:
    """Alias of :class:`ParquetFile` (``use_mmap=`` is accepted for ``memory_map=``)."""
    return ParquetFile(path, **kwargs)

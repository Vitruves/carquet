"""Shared fixtures for the carquet Python test suite."""

from __future__ import annotations

import datetime as dt
import decimal

import pytest

pa = pytest.importorskip("pyarrow")


def make_table(n: int = 1000) -> "pa.Table":
    """A table covering the common flat and nested types."""
    ids = pa.array(range(n), pa.int64())
    values = pa.array([i * 0.5 for i in range(n)], pa.float64())
    names = pa.array([None if i % 7 == 3 else f"name-{i}" for i in range(n)], pa.string())
    epoch = dt.datetime(2024, 1, 1)
    ts = pa.array([epoch + dt.timedelta(microseconds=1_000_003 * i) for i in range(n)], pa.timestamp("us"))
    lists = pa.array(
        [None if i % 11 == 5 else [j for j in range(i % 4)] for i in range(n)],
        pa.list_(pa.int32()),
    )
    struct = pa.StructArray.from_arrays(
        [
            pa.array([i % 100 for i in range(n)], pa.int32()),
            pa.array([None if i % 13 == 0 else f"s{i}" for i in range(n)], pa.string()),
        ],
        names=["a", "b"],
    )
    return pa.table({"id": ids, "value": values, "name": names, "ts": ts, "tags": lists, "point": struct})


def make_typed_table(n: int = 200) -> "pa.Table":
    """Logical types that need conversion or ARROW:schema restoration on read."""
    utc = dt.timezone.utc
    return pa.table(
        {
            "i8": pa.array([i % 100 - 50 for i in range(n)], pa.int8()),
            "u16": pa.array([i * 7 for i in range(n)], pa.uint16()),
            "u64": pa.array([2**63 + i for i in range(n)], pa.uint64()),
            "f32": pa.array([i / 3 for i in range(n)], pa.float32()),
            "f16": pa.array([float(i % 10) for i in range(n)], pa.float32()).cast(pa.float16()),
            "flag": pa.array([i % 3 == 0 for i in range(n)], pa.bool_()),
            "dec": pa.array([decimal.Decimal(i * 125 - 9000).scaleb(-2) for i in range(n)], pa.decimal128(9, 2)),
            "dec_big": pa.array([decimal.Decimal(10**30 + i).scaleb(-4) for i in range(n)], pa.decimal128(38, 4)),
            "date": pa.array([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)], pa.date32()),
            "ts_tz": pa.array([dt.datetime(2024, 3, 1, tzinfo=utc) + dt.timedelta(minutes=i) for i in range(n)],
                              pa.timestamp("ms", tz="Europe/Paris")),
            "ts_ns": pa.array([1_700_000_000_000_000_000 + i for i in range(n)], pa.timestamp("ns")),
            "t64": pa.array([dt.time(12, 0, i % 60) for i in range(n)], pa.time64("us")),
            "bin": pa.array([bytes([i % 256]) * (i % 5) for i in range(n)], pa.binary()),
            "fixed": pa.array([i.to_bytes(4, "little") for i in range(n)], pa.binary(4)),
            "lstr": pa.array([f"x{i}" for i in range(n)], pa.large_string()),
            "cat": pa.array([("a", "b", "c")[i % 3] for i in range(n)]).dictionary_encode(),
            "m": pa.array([[(f"k{i % 3}", i)] for i in range(n)], pa.map_(pa.string(), pa.int64())),
            "nested": pa.array([[{"x": i, "y": [i, i + 1]}] for i in range(n)],
                               pa.list_(pa.struct([("x", pa.int64()), ("y", pa.list_(pa.int16()))]))),
        }
    )


@pytest.fixture(scope="session")
def table() -> "pa.Table":
    return make_table()


@pytest.fixture(scope="session")
def typed_table() -> "pa.Table":
    return make_typed_table()


@pytest.fixture
def written(tmp_path, table):
    import carquet

    path = tmp_path / "sample.parquet"
    carquet.write_table(table, path, compression="zstd")
    return path


def assert_tables_equal(actual: "pa.Table", expected: "pa.Table") -> None:
    """Same schema and values; dictionary columns compare by value (their
    per-chunk dictionaries may be ordered differently)."""
    assert actual.schema.equals(expected.schema, check_metadata=False), (actual.schema, expected.schema)
    assert actual.num_rows == expected.num_rows
    for name in expected.column_names:
        a, b = actual.column(name), expected.column(name)
        if pa.types.is_dictionary(b.type):
            a, b = a.cast(b.type.value_type), b.cast(b.type.value_type)
        assert a.equals(b), name

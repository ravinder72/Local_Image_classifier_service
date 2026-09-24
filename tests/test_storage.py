"""Tests for the SQLite result repository."""

import pytest

from galaxeye.model import ClassifyResult, LandClass, ResultStatus
from galaxeye.storage import SQLiteResultRepository


def _result(id, tile_id="tile_001", label=LandClass.FOREST, confidence=0.9,
            status=ResultStatus.CONFIDENT, created_at="2026-09-24T12:00:00Z",
            model_version="resnet18-7cls-v0.1.0"):
    return ClassifyResult(
        id=id, tile_id=tile_id, label=label, confidence=confidence,
        status=status, created_at=created_at, model_version=model_version,
    )


def test_save_and_get_round_trip(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    result = _result("id-1")
    repo.save(result)
    got = repo.get("id-1")
    assert got == result


def test_get_missing_returns_none(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    assert repo.get("missing") is None


def test_query_by_class(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    repo.save(_result("1", label=LandClass.FOREST))
    repo.save(_result("2", label=LandClass.RIVER))
    results = repo.query(land_class=LandClass.FOREST)
    assert [r.id for r in results] == ["1"]


def test_query_by_min_confidence(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    repo.save(_result("1", confidence=0.9))
    repo.save(_result("2", confidence=0.5))
    results = repo.query(min_confidence=0.8)
    assert [r.id for r in results] == ["1"]


def test_query_combined_filters(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    repo.save(_result("1", label=LandClass.FOREST, confidence=0.9))
    repo.save(_result("2", label=LandClass.FOREST, confidence=0.5))
    repo.save(_result("3", label=LandClass.RIVER, confidence=0.9))
    results = repo.query(land_class=LandClass.FOREST, min_confidence=0.8)
    assert [r.id for r in results] == ["1"]


def test_query_returns_all_and_orders_by_created_desc(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    repo.save(_result("1", created_at="2026-01-01T00:00:00Z"))
    repo.save(_result("2", created_at="2026-02-01T00:00:00Z"))
    results = repo.query()
    assert [r.id for r in results] == ["2", "1"]


def test_query_respects_limit(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    for i in range(5):
        repo.save(_result(f"id-{i}"))
    assert len(repo.query(limit=3)) == 3


def test_duplicate_id_raises_storage_error(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    repo.save(_result("dup"))
    with pytest.raises(Exception):
        repo.save(_result("dup"))


def test_repository_schema_created_on_open(tmp_path):
    repo = SQLiteResultRepository(tmp_path / "test.db")
    repo.close()
    import sqlite3

    conn = sqlite3.connect(tmp_path / "test.db")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "results" in tables

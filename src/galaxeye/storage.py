"""Persistence layer: repository interface + SQLite implementation.

The repository interface is a clean boundary between the application and
storage, which keeps the SQLite implementation easy to swap and test.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from galaxeye.model import ClassifyResult, LandClass, ResultStatus


class StorageError(RuntimeError):
    """Raised when a storage operation fails."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id            TEXT PRIMARY KEY,
    tile_id       TEXT NOT NULL,
    label         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_label      ON results(label);
CREATE INDEX IF NOT EXISTS idx_results_confidence ON results(confidence);
CREATE INDEX IF NOT EXISTS idx_results_created    ON results(created_at);
"""


class ResultRepository(ABC):
    """Boundary between the application and persistence."""

    @abstractmethod
    def save(self, result: ClassifyResult) -> None: ...

    @abstractmethod
    def get(self, result_id: str) -> ClassifyResult | None: ...

    @abstractmethod
    def query(
        self,
        land_class: LandClass | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
    ) -> list[ClassifyResult]: ...


class SQLiteResultRepository(ResultRepository):
    """ResultRepository backed by a local SQLite database file."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI serves requests on worker threads
        # different from the one that constructs the connection.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def save(self, result: ClassifyResult) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO results
                    (id, tile_id, label, confidence, status, created_at, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.tile_id,
                    result.label.value,
                    result.confidence,
                    result.status.value,
                    result.created_at,
                    result.model_version,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to save result {result.id}: {exc}") from exc

    def get(self, result_id: str) -> ClassifyResult | None:
        try:
            row = self._conn.execute(
                "SELECT * FROM results WHERE id = ?", (result_id,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to fetch result {result_id}: {exc}") from exc
        return self._row_to_result(row) if row else None

    def query(
        self,
        land_class: LandClass | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
    ) -> list[ClassifyResult]:
        clauses: list[str] = []
        params: list[object] = []
        if land_class is not None:
            clauses.append("label = ?")
            params.append(land_class.value)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        try:
            rows = self._conn.execute(
                f"SELECT * FROM results {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Query failed: {exc}") from exc
        return [self._row_to_result(row) for row in rows]

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> ClassifyResult:
        return ClassifyResult(
            id=row["id"],
            tile_id=row["tile_id"],
            label=LandClass(row["label"]),
            confidence=row["confidence"],
            status=ResultStatus(row["status"]),
            created_at=row["created_at"],
            model_version=row["model_version"],
        )

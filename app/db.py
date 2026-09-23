"""SQLite persistence for jobs, input events and matched pairs."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

DB_PATH = os.environ.get("ALIGNMENT_DB_PATH", "/data/alignment.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    tolerance_us   INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    total_abs_diff INTEGER NOT NULL,
    candidate_edges INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    job_id          TEXT NOT NULL REFERENCES jobs(job_id),
    side            TEXT NOT NULL CHECK (side IN ('left', 'right')),
    event_id        TEXT NOT NULL,
    timestamp_iso   TEXT NOT NULL,
    timestamp_us    INTEGER NOT NULL,
    source_key      TEXT NOT NULL,
    payload_summary TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (job_id, side, event_id)
);

CREATE TABLE IF NOT EXISTS pairs (
    job_id       TEXT NOT NULL REFERENCES jobs(job_id),
    pair_ord     INTEGER NOT NULL,
    left_id      TEXT NOT NULL,
    right_id     TEXT NOT NULL,
    diff_us      INTEGER NOT NULL,
    abs_diff_us  INTEGER NOT NULL,
    PRIMARY KEY (job_id, pair_ord)
);

CREATE INDEX IF NOT EXISTS idx_pairs_job ON pairs(job_id);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, side);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""


def _connect() -> sqlite3.Connection:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: one connection per request, committed on success."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()

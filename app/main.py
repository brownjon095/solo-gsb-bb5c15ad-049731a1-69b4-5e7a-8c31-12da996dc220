"""FastAPI application: cross-source event alignment."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .alignment import AlignmentResult, align_events, source_pair_stats
from .db import get_conn, init_db
from .schemas import (
    AlignmentRequest,
    AlignmentResponse,
    AlignmentSummary,
    EventOut,
    HealthResponse,
    JobListItem,
    JobListResponse,
    MatchedPairOut,
    SourcePairStatOut,
    to_timestamp_us,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Cross-source Event Alignment API",
    description=(
        "Submit two groups of timestamped events; the service builds "
        "candidate edges inside a tolerance window and produces a "
        "minimum-total-difference, one-to-one matching with stable "
        "event-ID tie-breaks, keeping every unmatched event. Requests may "
        "optionally cap the number of matched pairs per "
        "(left source_key, right source_key) combination; the optimum is "
        "then computed exactly under those capacity constraints."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


def _get_db() -> Iterator[sqlite3.Connection]:
    with get_conn() as conn:
        yield conn


def _explanation(left_id: str, right_id: str, diff_us: int, abs_diff_us: int) -> str:
    if diff_us == 0:
        timing = "simultaneous"
    elif diff_us > 0:
        timing = f"'{left_id}' is {abs_diff_us} us later than '{right_id}'"
    else:
        timing = f"'{left_id}' is {abs_diff_us} us earlier than '{right_id}'"
    return (
        f"|t('{left_id}') - t('{right_id}')| = {abs_diff_us} us "
        f"(diff = t(left) - t(right) = {diff_us:+d} us; {timing})"
    )


def _build_response(conn: sqlite3.Connection, job_id: str) -> AlignmentResponse:
    job = conn.execute(
        "SELECT tolerance_us, total_abs_diff, candidate_edges FROM jobs WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")

    def fetch_events(side: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT event_id, timestamp_iso, timestamp_us, source_key, "
            "payload_summary FROM events WHERE job_id = ? AND side = ? "
            "ORDER BY event_id",
            (job_id, side),
        ).fetchall()

    left_rows = fetch_events("left")
    right_rows = fetch_events("right")
    left = [
        EventOut(
            id=r["event_id"],
            timestamp=r["timestamp_iso"],
            source_key=r["source_key"],
            payload_summary=r["payload_summary"],
        )
        for r in left_rows
    ]
    right = [
        EventOut(
            id=r["event_id"],
            timestamp=r["timestamp_iso"],
            source_key=r["source_key"],
            payload_summary=r["payload_summary"],
        )
        for r in right_rows
    ]

    pair_rows = conn.execute(
        "SELECT left_id, right_id, diff_us, abs_diff_us "
        "FROM pairs WHERE job_id = ? ORDER BY pair_ord",
        (job_id,),
    ).fetchall()
    pairs = [
        MatchedPairOut(
            left_id=r["left_id"],
            right_id=r["right_id"],
            diff_us=r["diff_us"],
            abs_diff_us=r["abs_diff_us"],
            explanation=_explanation(
                r["left_id"], r["right_id"], r["diff_us"], r["abs_diff_us"]
            ),
        )
        for r in pair_rows
    ]

    capacity_rows = conn.execute(
        "SELECT left_source_key, right_source_key, max_pairs "
        "FROM job_capacities WHERE job_id = ?",
        (job_id,),
    ).fetchall()
    capacities = {
        (r["left_source_key"], r["right_source_key"]): r["max_pairs"]
        for r in capacity_rows
    }
    stats = source_pair_stats(
        left_ids=[r["event_id"] for r in left_rows],
        left_times=[r["timestamp_us"] for r in left_rows],
        left_sources=[r["source_key"] for r in left_rows],
        right_ids=[r["event_id"] for r in right_rows],
        right_times=[r["timestamp_us"] for r in right_rows],
        right_sources=[r["source_key"] for r in right_rows],
        tolerance_us=job["tolerance_us"],
        pairs=[(r["left_id"], r["right_id"]) for r in pair_rows],
        capacities=capacities,
    )

    matched_left = {p.left_id for p in pairs}
    matched_right = {p.right_id for p in pairs}
    unmatched = {
        "left": [e for e in left if e.id not in matched_left],
        "right": [e for e in right if e.id not in matched_right],
    }

    return AlignmentResponse(
        job_id=job_id,
        tolerance_us=job["tolerance_us"],
        left=left,
        right=right,
        pairs=pairs,
        unmatched=unmatched,
        summary=AlignmentSummary(
            candidate_edges=job["candidate_edges"],
            matched_pairs=len(pairs),
            unmatched_left=len(unmatched["left"]),
            unmatched_right=len(unmatched["right"]),
            total_abs_diff_us=job["total_abs_diff"],
            capacity_rejected_edges=sum(s.rejected_edges for s in stats),
            source_pairs=[
                SourcePairStatOut(
                    left_source_key=s.left_source_key,
                    right_source_key=s.right_source_key,
                    capacity=s.capacity,
                    candidate_edges=s.candidate_edges,
                    matched_pairs=s.matched_pairs,
                    rejected_edges=s.rejected_edges,
                )
                for s in stats
            ],
        ),
    )


@app.post("/alignments", response_model=AlignmentResponse, status_code=201,
          summary="Create an alignment job")
def create_alignment(
    req: AlignmentRequest,
    conn: sqlite3.Connection = Depends(_get_db),
) -> Any:
    capacities = {
        (c.left_source_key, c.right_source_key): c.max_pairs
        for c in req.source_pair_capacities
    }
    result: AlignmentResult = align_events(
        left_ids=[e.id for e in req.left],
        left_times=[to_timestamp_us(e.timestamp) for e in req.left],
        right_ids=[e.id for e in req.right],
        right_times=[to_timestamp_us(e.timestamp) for e in req.right],
        tolerance_us=req.tolerance_us,
        left_sources=[e.source_key for e in req.left],
        right_sources=[e.source_key for e in req.right],
        capacities=capacities or None,
    )

    job_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO jobs (job_id, tolerance_us, created_at, total_abs_diff, "
        "candidate_edges) VALUES (?, ?, ?, ?, ?)",
        (
            job_id,
            req.tolerance_us,
            created_at,
            result.total_abs_diff_us,
            result.candidate_edge_count,
        ),
    )
    if capacities:
        conn.executemany(
            "INSERT INTO job_capacities (job_id, left_source_key, "
            "right_source_key, max_pairs) VALUES (?, ?, ?, ?)",
            [
                (job_id, left_key, right_key, max_pairs)
                for (left_key, right_key), max_pairs in sorted(capacities.items())
            ],
        )
    conn.executemany(
        "INSERT INTO events (job_id, side, event_id, timestamp_iso, "
        "timestamp_us, source_key, payload_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                job_id,
                "left",
                e.id,
                e.timestamp,
                to_timestamp_us(e.timestamp),
                e.source_key,
                e.payload_summary,
            )
            for e in req.left
        ]
        + [
            (
                job_id,
                "right",
                e.id,
                e.timestamp,
                to_timestamp_us(e.timestamp),
                e.source_key,
                e.payload_summary,
            )
            for e in req.right
        ],
    )
    conn.executemany(
        "INSERT INTO pairs (job_id, pair_ord, left_id, right_id, diff_us, "
        "abs_diff_us) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (job_id, ord_, p.left_id, p.right_id, p.diff_us, p.abs_diff_us)
            for ord_, p in enumerate(result.pairs)
        ],
    )

    return _build_response(conn, job_id)


@app.get("/alignments/{job_id}", response_model=AlignmentResponse,
         summary="Fetch a stored alignment job")
def get_alignment(
    job_id: str,
    conn: sqlite3.Connection = Depends(_get_db),
) -> Any:
    return _build_response(conn, job_id)


@app.get("/jobs", response_model=JobListResponse,
         summary="List recent alignment jobs (newest first, up to 100)")
def list_jobs(conn: sqlite3.Connection = Depends(_get_db)) -> Any:
    rows = conn.execute(
        "SELECT job_id, created_at, tolerance_us, total_abs_diff "
        "FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT 100"
    ).fetchall()
    counts = dict(
        conn.execute(
            "SELECT job_id, COUNT(*) FROM pairs GROUP BY job_id"
        ).fetchall()
    )
    return JobListResponse(
        items=[
            JobListItem(
                job_id=r["job_id"],
                created_at=r["created_at"],
                tolerance_us=r["tolerance_us"],
                matched_pairs=counts.get(r["job_id"], 0),
                total_abs_diff_us=r["total_abs_diff"],
            )
            for r in rows
        ]
    )


@app.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> Any:
    return HealthResponse(status="ok")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

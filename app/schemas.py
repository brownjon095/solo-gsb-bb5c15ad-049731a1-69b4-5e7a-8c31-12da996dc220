"""Pydantic models for the event-alignment API."""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _parse_iso(value: str) -> datetime:
    # datetime.fromisoformat in Python 3.11+ accepts "Z", but normalise it
    # explicitly for clarity.
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def to_timestamp_us(value: str | datetime) -> int:
    """Parse an ISO-8601 timestamp into integer microseconds (UTC epoch).

    Naive timestamps are assumed to be UTC. Uses integer arithmetic
    (``calendar.timegm``) so no float rounding occurs.
    """
    dt = value if isinstance(value, datetime) else _parse_iso(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return calendar.timegm(dt.timetuple()) * 1_000_000 + dt.microsecond


def to_utc_iso(value: str | datetime) -> str:
    """Normalise an ISO-8601 timestamp to UTC, e.g. ``2026-01-01T00:00:00+00:00``."""
    dt = value if isinstance(value, datetime) else _parse_iso(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class EventIn(BaseModel):
    """A client-submitted event."""

    id: str = Field(..., min_length=1, description="Unique ID within its group")
    timestamp: str | datetime = Field(
        ..., description="ISO-8601 timestamp, e.g. 2026-09-22T10:00:00Z"
    )
    source_key: str = Field(
        ..., min_length=1, description="Origin/source identifier"
    )
    payload_summary: str = Field(
        "", description="Short free-text summary of the payload"
    )

    @field_validator("timestamp")
    @classmethod
    def _normalise_timestamp(cls, v: str | datetime) -> str:
        return to_utc_iso(v)


class SourcePairCapacity(BaseModel):
    """Max matched pairs allowed for one (left, right) source-key pair."""

    left_source_key: str = Field(
        ..., min_length=1, description="source_key of left events"
    )
    right_source_key: str = Field(
        ..., min_length=1, description="source_key of right events"
    )
    max_pairs: int = Field(
        ..., ge=0, description="Maximum matched pairs for this source pair"
    )


class AlignmentRequest(BaseModel):
    """Two groups of events to align against each other."""

    tolerance_us: int = Field(
        ...,
        ge=0,
        description="Matching window; |t_a - t_b| <= tolerance_us",
    )
    left: list[EventIn] = Field(..., min_length=1)
    right: list[EventIn] = Field(..., min_length=1)
    source_pair_capacities: list[SourcePairCapacity] = Field(
        default_factory=list,
        description=(
            "Optional per-(left source_key, right source_key) caps on the "
            "number of matched pairs; combinations not listed are unlimited"
        ),
    )

    @model_validator(mode="after")
    def _unique_ids(self) -> "AlignmentRequest":
        for side_name, events in (("left", self.left), ("right", self.right)):
            ids = [e.id for e in events]
            dupes = {x for x in ids if ids.count(x) > 1}
            if dupes:
                raise ValueError(
                    f"duplicate event id(s) on {side_name}: {sorted(dupes)}"
                )
        return self

    @model_validator(mode="after")
    def _unique_capacity_pairs(self) -> "AlignmentRequest":
        keys = [
            (c.left_source_key, c.right_source_key)
            for c in self.source_pair_capacities
        ]
        dupes = {k for k in keys if keys.count(k) > 1}
        if dupes:
            raise ValueError(
                f"duplicate source-pair capacity entries: {sorted(dupes)}"
            )
        return self


class MatchedPairOut(BaseModel):
    left_id: str
    right_id: str
    diff_us: int = Field(
        ..., description="Signed difference: t(left) - t(right) in microseconds"
    )
    abs_diff_us: int = Field(..., description="|diff_us|")
    explanation: str


class EventOut(BaseModel):
    id: str
    timestamp: str
    source_key: str
    payload_summary: str = ""


class SourcePairStatOut(BaseModel):
    left_source_key: str
    right_source_key: str
    capacity: int | None = Field(
        ..., description="Declared max pairs for this combination; null = unlimited"
    )
    candidate_edges: int = Field(
        ..., description="Candidate edges between the two sources within tolerance"
    )
    matched_pairs: int = Field(
        ..., description="Pairs actually matched in this combination"
    )
    rejected_edges: int = Field(
        ...,
        description=(
            "Candidate edges denied by the capacity: "
            "max(0, candidate_edges - capacity); 0 when unlimited"
        ),
    )


class AlignmentSummary(BaseModel):
    candidate_edges: int
    matched_pairs: int
    unmatched_left: int
    unmatched_right: int
    total_abs_diff_us: int
    capacity_rejected_edges: int = Field(
        0, description="Total candidate edges rejected due to source-pair capacities"
    )
    source_pairs: list[SourcePairStatOut] = Field(
        default_factory=list,
        description="Per-(left source, right source) combination statistics",
    )


class AlignmentResponse(BaseModel):
    job_id: str
    tolerance_us: int
    left: list[EventOut]
    right: list[EventOut]
    pairs: list[MatchedPairOut]
    unmatched: dict[Literal["left", "right"], list[EventOut]]
    summary: AlignmentSummary


class JobListItem(BaseModel):
    job_id: str
    created_at: str
    tolerance_us: int
    matched_pairs: int
    total_abs_diff_us: int


class JobListResponse(BaseModel):
    items: list[JobListItem]


class HealthResponse(BaseModel):
    status: str

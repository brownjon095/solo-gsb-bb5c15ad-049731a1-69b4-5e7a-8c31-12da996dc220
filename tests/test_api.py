"""End-to-end API tests (run inside the container or with deps installed)."""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("ALIGNMENT_DB_PATH", os.path.join(tempfile.mkdtemp(), "test.db"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()      # triggers lifespan -> init_db

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def _payload(self, **overrides):
        payload = {
            "tolerance_us": 200_000,
            "left": [
                {
                    "id": "cam-a",
                    "timestamp": "2026-09-22T10:00:00Z",
                    "source_key": "camera-1",
                    "payload_summary": "person entered",
                },
                {
                    "id": "cam-b",
                    "timestamp": "2026-09-22T10:05:00Z",
                    "source_key": "camera-1",
                    "payload_summary": "person left",
                },
            ],
            "right": [
                {
                    "id": "rfid-1",
                    "timestamp": "2026-09-22T10:00:00.100000+00:00",
                    "source_key": "rfid-gate",
                    "payload_summary": "badge scan",
                },
                {
                    "id": "rfid-2",
                    "timestamp": "2026-09-22T10:05:00.050000+00:00",
                    "source_key": "rfid-gate",
                    "payload_summary": "badge scan",
                },
                {
                    "id": "rfid-lonely",
                    "timestamp": "2026-09-22T11:00:00Z",
                    "source_key": "rfid-gate",
                    "payload_summary": "orphan scan",
                },
            ],
        }
        payload.update(overrides)
        return payload

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_openapi_available(self):
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("/alignments", r.json()["paths"])

    def test_create_and_fetch_alignment(self):
        r = self.client.post("/alignments", json=self._payload())
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["tolerance_us"], 200_000)
        self.assertEqual(len(body["pairs"]), 2)
        self.assertEqual(body["summary"]["matched_pairs"], 2)
        self.assertEqual(body["summary"]["unmatched_right"], 1)
        self.assertEqual(body["summary"]["candidate_edges"], 2)

        by_left = {p["left_id"]: p for p in body["pairs"]}
        self.assertEqual(by_left["cam-a"]["right_id"], "rfid-1")
        self.assertEqual(by_left["cam-a"]["diff_us"], -100_000)
        self.assertEqual(by_left["cam-a"]["abs_diff_us"], 100_000)
        self.assertIn("t(left) - t(right) = -100000", by_left["cam-a"]["explanation"])
        self.assertEqual(by_left["cam-b"]["right_id"], "rfid-2")
        self.assertEqual(by_left["cam-b"]["abs_diff_us"], 50_000)

        self.assertEqual(
            [e["id"] for e in body["unmatched"]["right"]], ["rfid-lonely"]
        )
        self.assertEqual(body["unmatched"]["left"], [])
        self.assertEqual(body["summary"]["total_abs_diff_us"], 150_000)

        # Persisted: fetch by job id.
        job_id = body["job_id"]
        r2 = self.client.get(f"/alignments/{job_id}")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["pairs"], body["pairs"])

        # Listed.
        r3 = self.client.get("/jobs")
        self.assertEqual(r3.status_code, 200)
        ids = [item["job_id"] for item in r3.json()["items"]]
        self.assertIn(job_id, ids)

    def test_get_unknown_job_404(self):
        r = self.client.get("/alignments/does-not-exist")
        self.assertEqual(r.status_code, 404)

    def test_tie_break_is_id_stable_and_order_independent(self):
        def iso_us(t: int) -> str:
            return (
                f"1970-01-01T00:00:{t // 1_000_000:02d}.{t % 1_000_000:06d}Z"
            )

        def post(left, right):
            return self.client.post(
                "/alignments",
                json={
                    "tolerance_us": 110,
                    "left": [
                        {"id": eid, "timestamp": iso_us(t), "source_key": "L"}
                        for eid, t in left
                    ],
                    "right": [
                        {"id": eid, "timestamp": iso_us(t), "source_key": "R"}
                        for eid, t in right
                    ],
                },
            )

        # Both perfect matchings total 110 us; ID rule picks a0->b0.
        left = [("a0", 0), ("a1", 100)]
        right = [("b0", 100), ("b1", 110)]
        r1 = post(left, right)
        r2 = post(list(reversed(left)), list(reversed(right)))
        self.assertEqual(r1.status_code, 201, r1.text)
        pairs1 = {(p["left_id"], p["right_id"]) for p in r1.json()["pairs"]}
        pairs2 = {(p["left_id"], p["right_id"]) for p in r2.json()["pairs"]}
        self.assertEqual(pairs1, {("a0", "b0"), ("a1", "b1")})
        self.assertEqual(pairs1, pairs2)

    def test_validation_errors(self):
        # duplicate ids
        r = self.client.post(
            "/alignments",
            json={
                "tolerance_us": 100,
                "left": [
                    {"id": "x", "timestamp": "2026-01-01T00:00:00Z",
                     "source_key": "s"},
                    {"id": "x", "timestamp": "2026-01-01T00:00:01Z",
                     "source_key": "s"},
                ],
                "right": [
                    {"id": "y", "timestamp": "2026-01-01T00:00:00Z",
                     "source_key": "s"}
                ],
            },
        )
        self.assertEqual(r.status_code, 422)

        # negative tolerance
        r = self.client.post(
            "/alignments",
            json={
                "tolerance_us": -1,
                "left": [
                    {"id": "x", "timestamp": "2026-01-01T00:00:00Z",
                     "source_key": "s"}
                ],
                "right": [
                    {"id": "y", "timestamp": "2026-01-01T00:00:00Z",
                     "source_key": "s"}
                ],
            },
        )
        self.assertEqual(r.status_code, 422)

        # bad timestamp
        r = self.client.post(
            "/alignments",
            json={
                "tolerance_us": 0,
                "left": [
                    {"id": "x", "timestamp": "not-a-time", "source_key": "s"}
                ],
                "right": [
                    {"id": "y", "timestamp": "2026-01-01T00:00:00Z",
                     "source_key": "s"}
                ],
            },
        )
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()

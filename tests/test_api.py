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

    def test_source_capacity_constrains_global_match(self):
        payload = self._payload()
        payload["source_capacities"] = [
            {"left_source": "camera-1", "right_source": "rfid-gate",
             "max_matches": 1}
        ]
        r = self.client.post("/alignments", json=payload)
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        # Both candidate pairs belong to the capped combination, so only
        # one survives. Layer 2 (minimum total diff) keeps the tighter
        # edge cam-b/rfid-2 (50000 us) over cam-a/rfid-1 (100000 us).
        self.assertEqual(len(body["pairs"]), 1)
        self.assertEqual(body["pairs"][0]["left_id"], "cam-b")
        self.assertEqual(body["pairs"][0]["right_id"], "rfid-2")
        combos = body["summary"]["source_combos"]
        self.assertEqual(len(combos), 1)
        c = combos[0]
        self.assertEqual(c["left_source"], "camera-1")
        self.assertEqual(c["right_source"], "rfid-gate")
        self.assertEqual(c["capacity"], 1)
        self.assertEqual(c["candidate_edges"], 2)
        self.assertEqual(c["matched_pairs"], 1)
        self.assertEqual(c["rejected_by_capacity"], 1)
        self.assertEqual(body["summary"]["matched_pairs"], 1)
        self.assertEqual(body["summary"]["unmatched_left"], 1)
        self.assertEqual(body["summary"]["unmatched_right"], 2)

        # Persisted: refetch reports the same capacity summary.
        job_id = body["job_id"]
        again = self.client.get(f"/alignments/{job_id}").json()
        self.assertEqual(
            again["summary"]["source_combos"],
            body["summary"]["source_combos"],
        )
        self.assertEqual(again["pairs"], body["pairs"])

    def test_zero_capacity_forbids_combination(self):
        def iso(t):
            return f"2026-01-01T00:00:{t:02d}Z"

        payload = {
            "tolerance_us": 0,
            "left": [
                {"id": "a0", "timestamp": iso(0), "source_key": "L1"},
                {"id": "a1", "timestamp": iso(1), "source_key": "L2"},
            ],
            "right": [
                {"id": "b0", "timestamp": iso(0), "source_key": "R1"},
                {"id": "b1", "timestamp": iso(1), "source_key": "R2"},
            ],
            "source_capacities": [
                {"left_source": "L1", "right_source": "R1", "max_matches": 0}
            ],
        }
        r = self.client.post("/alignments", json=payload)
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        # At tol 0 only the diagonal pairs a0-b0 (banned) and a1-b1
        # (allowed) exist; the banned combination's events stay unmatched.
        self.assertEqual(
            {(p["left_id"], p["right_id"]) for p in body["pairs"]},
            {("a1", "b1")},
        )
        self.assertEqual(
            [e["id"] for e in body["unmatched"]["left"]], ["a0"]
        )
        self.assertEqual(
            [e["id"] for e in body["unmatched"]["right"]], ["b0"]
        )
        stats = {
            (c["left_source"], c["right_source"]): c
            for c in body["summary"]["source_combos"]
        }
        self.assertEqual(stats[("L1", "R1")]["capacity"], 0)
        self.assertEqual(stats[("L1", "R1")]["matched_pairs"], 0)
        self.assertEqual(stats[("L1", "R1")]["rejected_by_capacity"], 1)
        self.assertEqual(stats[("L2", "R2")]["capacity"], None)
        self.assertEqual(stats[("L2", "R2")]["matched_pairs"], 1)
        self.assertEqual(stats[("L2", "R2")]["rejected_by_capacity"], 0)

    def test_capacity_request_without_declaration_unchanged(self):
        # Legacy behaviour: no source_capacities field still gives the
        # full match and a summary with null capacities.
        r = self.client.post("/alignments", json=self._payload())
        self.assertEqual(r.status_code, 201, r.text)
        combos = r.json()["summary"]["source_combos"]
        self.assertEqual(len(combos), 1)
        self.assertIsNone(combos[0]["capacity"])
        self.assertEqual(combos[0]["matched_pairs"], 2)
        self.assertEqual(combos[0]["rejected_by_capacity"], 0)

    def test_duplicate_capacity_declaration_422(self):
        payload = self._payload()
        payload["source_capacities"] = [
            {"left_source": "camera-1", "right_source": "rfid-gate",
             "max_matches": 1},
            {"left_source": "camera-1", "right_source": "rfid-gate",
             "max_matches": 2},
        ]
        r = self.client.post("/alignments", json=payload)
        self.assertEqual(r.status_code, 422)

    def test_negative_capacity_422(self):
        payload = self._payload()
        payload["source_capacities"] = [
            {"left_source": "camera-1", "right_source": "rfid-gate",
             "max_matches": -1}
        ]
        r = self.client.post("/alignments", json=payload)
        self.assertEqual(r.status_code, 422)

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

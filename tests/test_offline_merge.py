import tempfile
import unittest
from pathlib import Path

from src.domain import Actor, ConflictError, PermissionDenied
from src.repository import SQLiteRepository
from src.rules import RuleEngine
from src.service import DomainService


class OfflineMergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = DomainService(SQLiteRepository(Path(self.tmp.name) / "test.db"), RuleEngine())
        self.admin = Actor("admin", "admin")
        self.field = Actor("field-1", "field")

    def tearDown(self):
        self.tmp.cleanup()

    def create(self, kind, data):
        return self.service.create(self.admin, kind, data)

    def merge(self, records, actor=None):
        return self.service.merge_offline(actor or self.field, records)

    @staticmethod
    def occupancy(source, record_id, refuge_id, worker_id):
        return {
            "source_id": source,
            "record_id": record_id,
            "recorded_at": "2026-09-27T10:00:00Z",
            "payload": {"type": "refuge_occupancy", "refuge_id": refuge_id, "worker_id": worker_id},
        }

    @staticmethod
    def restore(source, record_id, ventilation_id, gas_ppm=None):
        payload = {"type": "ventilation_restore", "ventilation_id": ventilation_id}
        if gas_ppm is not None:
            payload["gas_ppm"] = gas_ppm
        return {
            "source_id": source,
            "record_id": record_id,
            "recorded_at": "2026-09-27T10:05:00Z",
            "payload": payload,
        }

    def make_worker(self, name="Li Wei"):
        return self.create("worker", {"name": name, "location_code": "M-01", "team": "A"})

    def make_refuge(self, code, capacity):
        return self.create("refuge", {"location_code": code, "capacity": capacity})

    def make_ventilation(self, gas_ppm, threshold=100):
        sensor = self.create("sensor", {"location_code": "A-1", "gas_ppm": gas_ppm, "threshold_ppm": threshold})
        vent = self.create("ventilation", {"name": "fan-1", "area_code": "A-1", "capacity": 100})
        return sensor, vent

    def test_occupancy_applied_and_counted(self):
        refuge = self.make_refuge("R-1", 2)
        worker = self.make_worker()
        record = self.merge([self.occupancy("s1", "a1", refuge["id"], worker["id"])])[0]
        self.assertEqual(record["status"], "applied")
        refuge = self.service.get(refuge["id"])
        self.assertEqual(refuge["status"], "occupied")
        self.assertEqual(refuge["data"]["occupants"], [worker["id"]])
        self.assertEqual(refuge["data"]["occupied_count"], 1)

    def test_repeat_merge_does_not_double_occupy(self):
        refuge = self.make_refuge("R-1", 2)
        worker = self.make_worker()
        record = self.occupancy("s1", "a1", refuge["id"], worker["id"])
        first = self.merge([record])[0]
        second = self.merge([record])[0]
        self.assertEqual(first["id"], second["id"])
        # a different record id for the same worker and refuge is a consistent duplicate
        third = self.merge([self.occupancy("s1", "a2", refuge["id"], worker["id"])])[0]
        self.assertEqual(third["status"], "applied")
        refuge = self.service.get(refuge["id"])
        self.assertEqual(refuge["data"]["occupants"], [worker["id"]])
        self.assertEqual(refuge["data"]["occupied_count"], 1)

    def test_conflicting_records_for_same_worker_are_both_kept(self):
        worker = self.make_worker()
        r1 = self.make_refuge("R-1", 2)
        r2 = self.make_refuge("R-2", 2)
        first = self.merge([self.occupancy("s1", "a1", r1["id"], worker["id"])])[0]
        second = self.merge([self.occupancy("s1", "b1", r2["id"], worker["id"])])[0]
        self.assertEqual(second["status"], "conflict")
        first = self.service.get(first["id"])
        self.assertEqual(first["status"], "conflict")
        self.assertEqual(first["data"]["conflict_with"], [second["id"]])
        self.assertEqual(second["data"]["conflict_with"], [first["id"]])
        # both versions are kept; the first occupancy stays until resolved
        self.assertEqual(len(self.service.list("offline_record", status="conflict")), 2)
        self.assertEqual(self.service.get(r1["id"])["data"]["occupants"], [worker["id"]])
        self.assertIsNone(self.service.get(r2["id"])["data"].get("occupants"))

    def test_occupancy_beyond_capacity_is_rejected(self):
        refuge = self.make_refuge("R-1", 1)
        w1 = self.make_worker("Li Wei")
        w2 = self.make_worker("Wang Gang")
        first = self.merge([self.occupancy("s1", "a1", refuge["id"], w1["id"])])[0]
        self.assertEqual(first["status"], "applied")
        second = self.merge([self.occupancy("s1", "b1", refuge["id"], w2["id"])])[0]
        self.assertEqual(second["status"], "rejected")
        self.assertIn("capacity", second["data"]["reason"])
        refuge = self.service.get(refuge["id"])
        self.assertEqual(refuge["data"]["occupants"], [w1["id"]])
        self.assertEqual(refuge["data"]["occupied_count"], 1)

    def test_occupancy_with_unknown_worker_is_rejected(self):
        refuge = self.make_refuge("R-1", 2)
        record = self.merge([self.occupancy("s1", "a1", refuge["id"], "ghost-worker")])[0]
        self.assertEqual(record["status"], "rejected")
        self.assertIsNone(self.service.get(refuge["id"])["data"].get("occupants"))

    def test_restore_applies_when_gas_below_threshold(self):
        sensor, vent = self.make_ventilation(gas_ppm=50)
        self.service.transition(self.admin, vent["id"], "stop")
        record = self.merge([self.restore("s1", "v1", vent["id"], gas_ppm=50)])[0]
        self.assertEqual(record["status"], "applied")
        self.assertEqual(self.service.get(vent["id"])["status"], "running")
        again = self.merge([self.restore("s1", "v1", vent["id"], gas_ppm=50)])[0]
        self.assertEqual(again["id"], record["id"])

    def test_restore_waits_for_gas_and_reading_change_voids_request(self):
        sensor, vent = self.make_ventilation(gas_ppm=150)
        self.service.transition(self.admin, vent["id"], "stop")
        record = self.merge([self.restore("s1", "v1", vent["id"], gas_ppm=150)])[0]
        self.assertEqual(record["status"], "pending")
        self.assertEqual(self.service.get(vent["id"])["status"], "stopped")
        # gas reading changes before the request is processed -> voided
        self.service.transition(self.field, sensor["id"], "update_reading", {"gas_ppm": 160})
        record = self.service.get(record["id"])
        self.assertEqual(record["status"], "stale")
        # field confirmation while gas is still high -> stays pending
        record = self.service.transition(self.field, record["id"], "confirm")
        self.assertEqual(record["status"], "pending")
        # gas drops below threshold -> reading change voids it again
        self.service.transition(self.field, sensor["id"], "update_reading", {"gas_ppm": 80})
        self.assertEqual(self.service.get(record["id"])["status"], "stale")
        # field confirmation now lands the restore
        record = self.service.transition(self.field, record["id"], "confirm")
        self.assertEqual(record["status"], "applied")
        self.assertEqual(record["data"]["confirmed_by"], "field-1")
        self.assertEqual(self.service.get(vent["id"])["status"], "running")

    def test_restore_voided_at_merge_when_reading_changed(self):
        sensor, vent = self.make_ventilation(gas_ppm=150)
        self.service.transition(self.admin, vent["id"], "stop")
        # field noted gas at 100, but the official reading has since risen
        record = self.merge([self.restore("s1", "v1", vent["id"], gas_ppm=100)])[0]
        self.assertEqual(record["status"], "stale")
        self.assertEqual(self.service.get(vent["id"])["status"], "stopped")

    def test_restore_with_unknown_ventilation_is_rejected(self):
        record = self.merge([self.restore("s1", "v1", "ghost-vent")])[0]
        self.assertEqual(record["status"], "rejected")

    def test_unresolved_conflict_blocks_incident_close(self):
        incident = self.create("incident", {"area_code": "M-01", "severity": "high", "summary": "gas leak"})
        for action in ("begin_evacuation", "search", "stabilize", "recover"):
            incident = self.service.transition(self.admin, incident["id"], action)
        worker = self.make_worker()
        r1 = self.make_refuge("R-1", 2)
        r2 = self.make_refuge("R-2", 2)
        self.merge([self.occupancy("s1", "a1", r1["id"], worker["id"])])
        self.merge([self.occupancy("s1", "b1", r2["id"], worker["id"])])
        with self.assertRaises(ConflictError):
            self.service.transition(self.admin, incident["id"], "close", {"summary": "done"})
        conflicts = self.service.list("offline_record", status="conflict")
        self.assertEqual(len(conflicts), 2)
        second = [c for c in conflicts if c["data"]["record_id"] == "b1"][0]
        discarded = self.service.transition(self.admin, second["id"], "resolve_discard")
        self.assertEqual(discarded["status"], "superseded")
        incident = self.service.transition(self.admin, incident["id"], "close", {"summary": "done"})
        self.assertEqual(incident["status"], "closed")

    def test_resolve_apply_lands_chosen_version(self):
        worker = self.make_worker()
        r1 = self.make_refuge("R-1", 2)
        r2 = self.make_refuge("R-2", 2)
        first = self.merge([self.occupancy("s1", "a1", r1["id"], worker["id"])])[0]
        second = self.merge([self.occupancy("s1", "b1", r2["id"], worker["id"])])[0]
        second = self.service.transition(self.admin, second["id"], "resolve_apply")
        self.assertEqual(second["status"], "applied")
        self.assertEqual(self.service.get(first["id"])["status"], "superseded")
        r1 = self.service.get(r1["id"])
        r2 = self.service.get(r2["id"])
        self.assertEqual(r1["data"]["occupants"], [])
        self.assertEqual(r1["status"], "available")
        self.assertEqual(r2["data"]["occupants"], [worker["id"]])
        self.assertEqual(r2["status"], "occupied")

    def test_resolve_discard_keeps_first_version(self):
        worker = self.make_worker()
        r1 = self.make_refuge("R-1", 2)
        r2 = self.make_refuge("R-2", 2)
        first = self.merge([self.occupancy("s1", "a1", r1["id"], worker["id"])])[0]
        second = self.merge([self.occupancy("s1", "b1", r2["id"], worker["id"])])[0]
        second = self.service.transition(self.admin, second["id"], "resolve_discard")
        self.assertEqual(second["status"], "superseded")
        self.assertEqual(self.service.get(first["id"])["status"], "applied")
        self.assertEqual(self.service.get(r1["id"])["data"]["occupants"], [worker["id"]])
        self.assertIsNone(self.service.get(r2["id"])["data"].get("occupants"))

    def test_merge_requires_field_role(self):
        refuge = self.make_refuge("R-1", 2)
        worker = self.make_worker()
        with self.assertRaises(PermissionDenied):
            self.merge([self.occupancy("s1", "a1", refuge["id"], worker["id"])], actor=Actor("viewer", "viewer"))


if __name__ == "__main__":
    unittest.main()

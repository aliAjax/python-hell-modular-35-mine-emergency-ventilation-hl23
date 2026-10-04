import tempfile
import unittest
from pathlib import Path

from src.domain import Actor, ConflictError
from src.repository import SQLiteRepository
from src.rules import RuleEngine
from src.service import DomainService


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = DomainService(SQLiteRepository(Path(self.tmp.name) / "test.db"), RuleEngine())
        self.actor = Actor("admin", "admin")
        self._seq = 0

    def tearDown(self):
        self.tmp.cleanup()

    def create(self, kind, data):
        return self.service.create(self.actor, kind, data)

    def act(self, entity, action, data=None):
        return self.service.transition(self.actor, entity["id"], action, data or {})

    def merge(self, *payloads):
        records = []
        for p in payloads:
            self._seq += 1
            records.append({
                "source_id": "field",
                "record_id": str(self._seq),
                "recorded_at": "2026-09-27T10:00:00Z",
                "payload": p,
            })
        return self.service.merge_offline(self.actor, records)

    def merge_record(self, source_id, record_id, payload):
        record = {
            "source_id": source_id,
            "record_id": record_id,
            "recorded_at": "2026-09-27T10:00:00Z",
            "payload": payload,
        }
        return self.service.merge_offline(self.actor, [record])

    def test_occupancy_applied_and_capacity_rejected(self):
        refuge = self.create("refuge", {"location_code": "R-1", "capacity": 2})
        w1 = self.create("worker", {"name": "Li Wei", "location_code": "R-1", "team": "A"})
        w2 = self.create("worker", {"name": "Wang Fang", "location_code": "R-1", "team": "A"})
        w3 = self.create("worker", {"name": "Zhao Min", "location_code": "R-1", "team": "B"})

        results = self.merge(
            {"type": "occupancy", "person_id": w1["id"], "refuge_id": refuge["id"], "action": "occupy"},
            {"type": "occupancy", "person_id": w2["id"], "refuge_id": refuge["id"], "action": "occupy"},
            {"type": "occupancy", "person_id": w3["id"], "refuge_id": refuge["id"], "action": "occupy"},
        )
        self.assertEqual(results[0]["status"], "applied")
        self.assertEqual(results[1]["status"], "applied")
        self.assertEqual(results[2]["status"], "rejected")
        self.assertIn("capacity", results[2]["data"].get("reason", ""))
        self.assertEqual(self.service.get(refuge["id"])["data"]["occupied"], 2)

    def test_duplicate_merge_does_not_double_occupy(self):
        refuge = self.create("refuge", {"location_code": "R-1", "capacity": 2})
        w1 = self.create("worker", {"name": "Li Wei", "location_code": "R-1", "team": "A"})
        payload = {"type": "occupancy", "person_id": w1["id"], "refuge_id": refuge["id"], "action": "occupy"}
        first = self.merge_record("field", "dup-1", payload)
        second = self.merge_record("field", "dup-1", payload)
        self.assertEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(second[0]["status"], "applied")
        self.assertEqual(self.service.get(refuge["id"])["data"]["occupied"], 1)
        self.assertEqual(len(self.service.list("offline_record")), 1)

    def test_same_person_mismatch_marks_conflict(self):
        r1 = self.create("refuge", {"location_code": "R-1", "capacity": 4})
        r2 = self.create("refuge", {"location_code": "R-2", "capacity": 4})
        w1 = self.create("worker", {"name": "Li Wei", "location_code": "R-1", "team": "A"})

        results = self.merge(
            {"type": "occupancy", "person_id": w1["id"], "refuge_id": r1["id"], "action": "occupy"},
            {"type": "occupancy", "person_id": w1["id"], "refuge_id": r2["id"], "action": "occupy"},
        )
        # 处理第二条时第一条已在 DB 中被改为 conflict，需重新读取。
        self.assertEqual(self.service.get(results[0]["id"])["status"], "conflict")
        self.assertEqual(results[1]["status"], "conflict")
        # 冲突的第二版没有落地到 r2。
        self.assertEqual(self.service.get(r2["id"])["data"].get("occupied", 0), 0)
        conflicts = self.service.list("offline_record", status="conflict")
        self.assertEqual(len(conflicts), 2)

    def test_gas_change_voids_pending_recovery_then_confirm_applies(self):
        sensor = self.create("sensor", {"location_code": "A", "gas_ppm": 120, "threshold_ppm": 80})
        vent = self.create("ventilation", {"name": "fan-1", "area_code": "A", "capacity": 100})
        self.act(vent, "stop")

        pending = self.merge({"type": "recovery", "ventilation_id": vent["id"]})
        self.assertEqual(pending[0]["status"], "pending")

        # 气体读数变化：待处理的恢复申请作废。
        changed = self.merge({"type": "gas", "sensor_id": sensor["id"], "value": 40})
        self.assertEqual(changed[0]["status"], "applied")
        self.assertEqual(self.service.get(pending[0]["id"])["status"], "voided")

        # 现场确认后才落地：气体已降到阈值以下，恢复运行。
        confirmed = self.merge({"type": "recovery_confirm", "ventilation_id": vent["id"]})
        self.assertEqual(confirmed[0]["status"], "applied")
        self.assertEqual(self.service.get(vent["id"])["status"], "running")

    def test_recovery_confirm_rejected_while_gas_unsafe(self):
        self.create("sensor", {"location_code": "A", "gas_ppm": 120, "threshold_ppm": 80})
        vent = self.create("ventilation", {"name": "fan-1", "area_code": "A", "capacity": 100})
        self.act(vent, "stop")

        self.merge({"type": "recovery", "ventilation_id": vent["id"]})
        confirmed = self.merge({"type": "recovery_confirm", "ventilation_id": vent["id"]})
        self.assertEqual(confirmed[0]["status"], "rejected")
        self.assertIn("gas", confirmed[0]["data"].get("reason", ""))
        self.assertEqual(self.service.get(vent["id"])["status"], "stopped")

    def test_unresolved_conflict_blocks_incident_close(self):
        r1 = self.create("refuge", {"location_code": "R-1", "capacity": 4})
        r2 = self.create("refuge", {"location_code": "R-2", "capacity": 4})
        self.create("worker", {"name": "Li Wei", "location_code": "R-1", "team": "A"})
        self.create("ventilation", {"name": "fan-1", "area_code": "A", "capacity": 100})
        incident = self.create("incident", {"area_code": "A", "severity": "critical", "summary": "gas leak"})
        for action in ("begin_evacuation", "search", "stabilize", "recover"):
            incident = self.act(incident, action)

        w1 = self.service.list("worker")[0]
        self.merge(
            {"type": "occupancy", "person_id": w1["id"], "refuge_id": r1["id"], "action": "occupy"},
            {"type": "occupancy", "person_id": w1["id"], "refuge_id": r2["id"], "action": "occupy"},
        )
        with self.assertRaises(ConflictError):
            self.act(incident, "close", {"summary": "done"})

        # 冲突处理完后，事件关闭不再被挡。
        resolved = self.merge({"type": "conflict_resolve", "person_id": w1["id"]})
        self.assertEqual(resolved[0]["status"], "applied")
        incident = self.act(incident, "close", {"summary": "all clear"})
        self.assertEqual(incident["status"], "closed")


if __name__ == "__main__":
    unittest.main()

import hashlib
from uuid import uuid4

from .audit import AuditTrail
from .domain import ConflictError, NotFoundError, PermissionDenied, ValidationError
from .repository import utcnow
from .rules import RuleEngine

OCCUPANCY_RECORD = "refuge_occupancy"
RESTORE_RECORD = "ventilation_restore"


class DomainService:
    def __init__(self, repository, rules=None):
        self.repository = repository
        self.rules = rules or RuleEngine()
        self.audit = AuditTrail(repository)

    def _lookup(self, kind, field, value):
        return self.repository.find_entities(self.rules.normalize_kind(kind), field, value)

    def health(self):
        return {"status": "ok" if self.repository.ping() else "error"}

    def create(self, actor, kind, data, idempotency_key=None):
        kind = self.rules.normalize_kind(kind)
        payload = dict(data or {})
        if idempotency_key:
            existing = self.repository.get_idempotency(actor.user_id, idempotency_key)
            if existing:
                entity = self.repository.get_entity(existing)
                if entity:
                    return entity
        self.rules.validate_create(actor, kind, payload, self._lookup)
        entity_id = str(payload.pop("id", "") or uuid4())
        if self.repository.get_entity(entity_id):
            raise ConflictError("entity already exists: " + entity_id)
        status = self.rules.initial_status(kind, payload)
        entity = self.repository.create_entity(entity_id, kind, status, payload, actor.user_id)
        self.audit.record(entity_id, actor, "create", None, status, {"kind": kind})
        if idempotency_key:
            self.repository.save_idempotency(actor.user_id, idempotency_key, entity_id)
        return entity

    def transition(self, actor, entity_id, action, data=None, expected_version=None):
        entity = self.repository.get_entity(entity_id)
        if not entity:
            raise NotFoundError("entity not found: " + entity_id)
        expected = int(expected_version) if expected_version is not None else entity["version"]
        next_status, patch = self.rules.validate_transition(
            actor, entity, action, dict(data or {}), self._lookup
        )
        merged = dict(entity["data"])
        merged.update(patch)
        updated = self.repository.update_entity(entity_id, expected, next_status, merged)
        self.audit.record(
            entity_id,
            actor,
            action,
            entity["status"],
            updated["status"],
            {"patch": patch},
        )
        return self._post_transition(actor, entity, updated, action)

    def merge_offline(self, actor, records):
        """Merge field records by a stable (source_id, record_id) identity.

        Known payload types are verified against personnel and equipment before
        they may touch the official state: refuge occupancy records are checked
        against the worker roster, conflicting records for the same worker and
        the chamber capacity; ventilation restore requests are checked against
        current gas readings. Re-merging the same (source_id, record_id) is
        idempotent and never applies twice.
        """
        if not isinstance(records, list):
            raise ValidationError("records must be a list")
        created = []
        for raw in records:
            if not isinstance(raw, dict):
                raise ValidationError("each offline record must be an object")
            source_id = str(raw.get("source_id", "")).strip()
            record_id = str(raw.get("record_id", "")).strip()
            if not source_id or not record_id:
                raise ValidationError("source_id and record_id are required")
            digest = hashlib.sha256((source_id + "\0" + record_id).encode("utf-8")).hexdigest()[:32]
            entity_id = "offline-" + digest
            existing = self.repository.get_entity(entity_id)
            if existing:
                created.append(existing)
                continue
            payload = dict(raw)
            self.rules.validate_create(actor, "offline_record", payload, self._lookup)
            record_type = (payload.get("payload") or {}).get("type")
            initial = "pending" if record_type in (OCCUPANCY_RECORD, RESTORE_RECORD) \
                else self.rules.initial_status("offline_record", payload)
            entity = self.repository.create_entity(
                entity_id,
                "offline_record",
                initial,
                payload,
                actor.user_id,
            )
            self.audit.record(entity_id, actor, "merge_offline", None, entity["status"], {"source_id": source_id, "record_id": record_id})
            if record_type == OCCUPANCY_RECORD:
                entity = self._process_occupancy(actor, entity)
            elif record_type == RESTORE_RECORD:
                entity = self._process_restore(actor, entity)
            created.append(entity)
        return created

    # ---- offline record verification pipeline ----

    def _mark_record(self, actor, record, status, updates=None, action=None):
        data = dict(record["data"])
        if updates:
            data.update(updates)
        updated = self.repository.update_entity(record["id"], record["version"], status, data)
        self.audit.record(
            record["id"],
            actor,
            action or ("offline_" + status),
            record["status"],
            status,
            {"data": updates or {}},
        )
        return updated

    def _post_transition(self, actor, previous, updated, action):
        kind = updated["kind"]
        if kind == "sensor":
            before = (previous["data"] or {}).get("gas_ppm")
            after = (updated["data"] or {}).get("gas_ppm")
            try:
                changed = float(before) != float(after)
            except (TypeError, ValueError):
                changed = before != after
            if changed:
                self._void_pending_restores(actor, updated)
            return updated
        if kind == "offline_record":
            if action == "confirm":
                return self._after_confirm(actor, updated)
            if action == "resolve_apply":
                return self._after_resolve_apply(actor, updated)
            if action == "resolve_discard":
                return self._after_resolve_discard(actor, updated)
        return updated

    # ---- refuge occupancy ----

    def _occupancy_peers(self, worker_id, exclude_id):
        peers = []
        for record in self.repository.list_entities(kind="offline_record"):
            if record["id"] == exclude_id or record["status"] not in ("pending", "applied", "conflict"):
                continue
            payload = record["data"].get("payload") or {}
            if payload.get("type") == OCCUPANCY_RECORD and payload.get("worker_id") == worker_id:
                peers.append(record)
        return peers

    def _process_occupancy(self, actor, record):
        payload = record["data"]["payload"]
        worker_id = payload.get("worker_id")
        refuge_id = payload.get("refuge_id")
        worker = self.repository.get_entity(worker_id)
        refuge = self.repository.get_entity(refuge_id)
        if not worker or worker["kind"] != "worker":
            return self._mark_record(actor, record, "rejected", {"reason": "worker not found: " + str(worker_id)})
        if not refuge or refuge["kind"] != "refuge":
            return self._mark_record(actor, record, "rejected", {"reason": "refuge not found: " + str(refuge_id)})
        if refuge["status"] not in ("available", "occupied"):
            return self._mark_record(actor, record, "rejected", {"reason": "refuge is not available for occupancy"})
        peers = self._occupancy_peers(worker_id, record["id"])
        conflicts = [p for p in peers if (p["data"].get("payload") or {}).get("refuge_id") != refuge_id]
        if conflicts:
            reason = "conflicting occupancy records for worker: " + str(worker_id)
            for peer in conflicts:
                linked = set(peer["data"].get("conflict_with") or [])
                linked.add(record["id"])
                self._mark_record(actor, peer, "conflict", {"conflict_with": sorted(linked), "reason": reason})
            return self._mark_record(
                actor,
                record,
                "conflict",
                {"conflict_with": sorted(p["id"] for p in conflicts), "reason": reason},
            )
        occupants = refuge["data"].get("occupants") or []
        capacity = float(refuge["data"].get("capacity", 0))
        if worker_id not in occupants and len(occupants) + 1 > capacity:
            return self._mark_record(actor, record, "rejected", {"reason": "refuge occupancy would exceed capacity"})
        self._apply_occupancy(actor, refuge, worker_id, record)
        return self._mark_record(actor, record, "applied", {"applied": True, "applied_at": utcnow()})

    def _apply_occupancy(self, actor, refuge, worker_id, record):
        occupants = list(refuge["data"].get("occupants") or [])
        if worker_id not in occupants:
            occupants.append(worker_id)
        data = dict(refuge["data"])
        data["occupants"] = occupants
        data["occupied_count"] = len(occupants)
        status = refuge["status"]
        if occupants and status == "available":
            status = "occupied"
        updated = self.repository.update_entity(refuge["id"], refuge["version"], status, data)
        self.audit.record(
            refuge["id"],
            actor,
            "occupy_offline",
            refuge["status"],
            status,
            {"worker_id": worker_id, "record_id": record["id"]},
        )
        return updated

    def _remove_occupancy(self, actor, refuge, worker_id, record):
        occupants = [item for item in (refuge["data"].get("occupants") or []) if item != worker_id]
        data = dict(refuge["data"])
        data["occupants"] = occupants
        data["occupied_count"] = len(occupants)
        status = refuge["status"]
        if not occupants and status == "occupied":
            status = "available"
        updated = self.repository.update_entity(refuge["id"], refuge["version"], status, data)
        self.audit.record(
            refuge["id"],
            actor,
            "release_offline",
            refuge["status"],
            status,
            {"worker_id": worker_id, "record_id": record["id"]},
        )
        return updated

    def _after_resolve_apply(self, actor, record):
        payload = record["data"].get("payload") or {}
        worker_id = payload.get("worker_id")
        for peer_id in record["data"].get("conflict_with") or []:
            peer = self.repository.get_entity(peer_id)
            if not peer or peer["status"] in ("superseded", "rejected"):
                continue
            peer_data = peer["data"]
            peer_payload = peer_data.get("payload") or {}
            if peer_data.get("applied") and peer_payload.get("type") == OCCUPANCY_RECORD:
                peer_refuge = self.repository.get_entity(peer_payload.get("refuge_id"))
                if peer_refuge:
                    self._remove_occupancy(actor, peer_refuge, peer_payload.get("worker_id"), peer)
            self._mark_record(actor, peer, "superseded", {"applied": False, "conflict_with": []})
        refuge = self.repository.get_entity(payload.get("refuge_id"))
        if not refuge:
            raise NotFoundError("refuge not found: " + str(payload.get("refuge_id")))
        self._apply_occupancy(actor, refuge, worker_id, record)
        return self._mark_record(
            actor,
            record,
            "applied",
            {"applied": True, "applied_at": utcnow(), "conflict_with": []},
        )

    def _after_resolve_discard(self, actor, record):
        data = record["data"]
        payload = data.get("payload") or {}
        if data.get("applied") and payload.get("type") == OCCUPANCY_RECORD:
            refuge = self.repository.get_entity(payload.get("refuge_id"))
            if refuge:
                self._remove_occupancy(actor, refuge, payload.get("worker_id"), record)
        for peer_id in data.get("conflict_with") or []:
            peer = self.repository.get_entity(peer_id)
            if not peer or peer["status"] != "conflict":
                continue
            remaining = [item for item in (peer["data"].get("conflict_with") or []) if item != record["id"]]
            if remaining:
                self._mark_record(actor, peer, "conflict", {"conflict_with": remaining})
            elif peer["data"].get("applied"):
                self._mark_record(actor, peer, "applied", {"conflict_with": []})
            else:
                self._reprocess_occupancy(actor, peer)
        return self._mark_record(actor, record, "superseded", {"applied": False, "conflict_with": []})

    def _reprocess_occupancy(self, actor, record):
        payload = record["data"].get("payload") or {}
        refuge = self.repository.get_entity(payload.get("refuge_id"))
        worker_id = payload.get("worker_id")
        updates = {"conflict_with": []}
        if not refuge or refuge["status"] not in ("available", "occupied"):
            updates["reason"] = "refuge is not available for occupancy"
            return self._mark_record(actor, record, "rejected", updates)
        occupants = refuge["data"].get("occupants") or []
        capacity = float(refuge["data"].get("capacity", 0))
        if worker_id not in occupants and len(occupants) + 1 > capacity:
            updates["reason"] = "refuge occupancy would exceed capacity"
            return self._mark_record(actor, record, "rejected", updates)
        self._apply_occupancy(actor, refuge, worker_id, record)
        updates.update({"applied": True, "applied_at": utcnow()})
        return self._mark_record(actor, record, "applied", updates)

    # ---- ventilation restore ----

    def _resolve_sensor(self, ventilation, sensor_id=None):
        if sensor_id:
            sensor = self.repository.get_entity(sensor_id)
            if sensor and sensor["kind"] == "sensor":
                return sensor
            return None
        area = ventilation["data"].get("area_code")
        for sensor in self.repository.list_entities(kind="sensor"):
            if sensor["data"].get("location_code") == area:
                return sensor
        return None

    def _process_restore(self, actor, record):
        payload = record["data"]["payload"]
        ventilation = self.repository.get_entity(payload.get("ventilation_id"))
        if not ventilation or ventilation["kind"] != "ventilation":
            return self._mark_record(actor, record, "rejected", {"reason": "ventilation not found: " + str(payload.get("ventilation_id"))})
        sensor = self._resolve_sensor(ventilation, payload.get("sensor_id"))
        if not sensor:
            return self._mark_record(actor, record, "rejected", {"reason": "no gas sensor available for ventilation"})
        gas = float(sensor["data"].get("gas_ppm", 0))
        baseline = payload.get("gas_ppm")
        if baseline is None:
            baseline = gas
        updates = {"sensor_id": sensor["id"], "gas_baseline": baseline}
        if float(baseline) != gas:
            updates["reason"] = "gas reading changed since the restore request was recorded"
            return self._mark_record(actor, record, "stale", updates)
        return self._evaluate_restore(actor, record, ventilation, sensor, updates)

    def _evaluate_restore(self, actor, record, ventilation, sensor, updates):
        updates = dict(updates)
        gas = float(sensor["data"].get("gas_ppm", 0))
        threshold = float(sensor["data"].get("threshold_ppm", 1))
        if ventilation["status"] != "running" and gas >= threshold:
            return self._mark_record(actor, record, "pending", updates)
        if ventilation["status"] in ("stopped", "degraded"):
            data = dict(ventilation["data"])
            data["restored_at"] = utcnow()
            data["restore_record_id"] = record["id"]
            self.repository.update_entity(ventilation["id"], ventilation["version"], "running", data)
            self.audit.record(
                ventilation["id"],
                actor,
                "restore_offline",
                ventilation["status"],
                "running",
                {"record_id": record["id"], "gas_ppm": gas},
            )
        updates.update({"applied": True, "applied_at": utcnow(), "gas_at_apply": gas})
        return self._mark_record(actor, record, "applied", updates)

    def _after_confirm(self, actor, record):
        payload = record["data"].get("payload") or {}
        ventilation = self.repository.get_entity(payload.get("ventilation_id"))
        if not ventilation or ventilation["kind"] != "ventilation":
            return self._mark_record(actor, record, "rejected", {"reason": "ventilation not found: " + str(payload.get("ventilation_id"))})
        sensor = self._resolve_sensor(ventilation, record["data"].get("sensor_id") or payload.get("sensor_id"))
        if not sensor:
            return self._mark_record(actor, record, "rejected", {"reason": "no gas sensor available for ventilation"})
        gas = float(sensor["data"].get("gas_ppm", 0))
        updates = {
            "sensor_id": sensor["id"],
            "gas_baseline": gas,
            "confirmed_by": actor.user_id,
            "confirmed_at": utcnow(),
        }
        return self._evaluate_restore(actor, record, ventilation, sensor, updates)

    def _void_pending_restores(self, actor, sensor):
        for record in self.repository.list_entities(kind="offline_record", status="pending"):
            payload = record["data"].get("payload") or {}
            if payload.get("type") != RESTORE_RECORD:
                continue
            if record["data"].get("sensor_id") != sensor["id"]:
                continue
            self._mark_record(
                actor,
                record,
                "stale",
                {"reason": "gas reading changed before the restore was processed"},
                action="void_restore",
            )

    # ---- reads ----

    def get(self, entity_id):
        entity = self.repository.get_entity(entity_id)
        if not entity:
            raise NotFoundError("entity not found: " + entity_id)
        return entity

    def list(self, kind=None, status=None):
        if kind:
            kind = self.rules.normalize_kind(kind)
        return self.repository.list_entities(kind=kind, status=status)

    def audit_log(self, entity_id=None):
        return self.repository.list_audit(entity_id=entity_id)

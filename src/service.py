import hashlib
from uuid import uuid4

from .audit import AuditTrail
from .domain import ConflictError, NotFoundError, PermissionDenied, ValidationError
from .repository import utcnow
from .rules import RuleEngine


class DomainService:
    def __init__(self, repository, rules=None):
        self.repository = repository
        self.rules = rules or RuleEngine()
        self.audit = AuditTrail(repository)
        self._seq_value = None

    def _next_seq(self):
        """单调递增序号，用于同一批记录内的先后排序。

        同一批现场记录的 recorded_at 往往相同，无法据其判断先后，
        因此用序号保证后到的占用/释放能覆盖先前的状态。
        """
        if self._seq_value is None:
            max_seq = 0
            for record in self.repository.list_entities(kind="offline_record"):
                max_seq = max(max_seq, int(record["data"].get("_seq", 0)))
            self._seq_value = max_seq
        self._seq_value += 1
        return self._seq_value

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
        return updated

    # ------------------------------------------------------------------
    # 离线记录合并
    # ------------------------------------------------------------------

    def merge_offline(self, actor, records):
        """Merge field records by a stable (source_id, record_id) identity.

        合并先核对人员与设备： occupancy 记录核对人员与硐室并做容量校验，
        recovery 记录只登记为待处理，气体读数变化会作废尚未处理的恢复申请，
        需现场确认后才落地。同一个人的记录对不上时两版都保留并标记冲突。
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
                # 幂等：同一条现场记录只处理一次，重复合并不会重复占用。
                created.append(existing)
                continue
            payload = dict(raw)
            self.rules.validate_create(actor, "offline_record", payload, self._lookup)
            record_data = dict(payload)
            record_data["_seq"] = self._next_seq()
            entity = self.repository.create_entity(
                entity_id,
                "offline_record",
                self.rules.initial_status("offline_record", payload),
                record_data,
                actor.user_id,
            )
            self.audit.record(entity_id, actor, "merge_offline", None, entity["status"], {"source_id": source_id, "record_id": record_id})
            entity = self._apply_offline_payload(actor, entity)
            created.append(entity)
        return created

    def _apply_offline_payload(self, actor, entity):
        payload = entity["data"].get("payload", {})
        ptype = payload.get("type")
        if ptype == "occupancy":
            return self._apply_occupancy(actor, entity, payload)
        if ptype == "recovery":
            return self._apply_recovery_request(actor, entity, payload)
        if ptype == "recovery_confirm":
            return self._apply_recovery_confirm(actor, entity, payload)
        if ptype == "gas":
            return self._apply_gas(actor, entity, payload)
        if ptype == "conflict_resolve":
            return self._apply_conflict_resolve(actor, entity, payload)
        return entity

    def _set_record_status(self, entity, status, **extra):
        data = dict(entity["data"])
        data.update(extra)
        return self.repository.update_entity(entity["id"], entity["version"], status, data)

    def _find_person(self, payload):
        pid = payload.get("person_id")
        if pid:
            person = self.repository.get_entity(pid)
            if person and person["kind"] == "worker":
                return person
        name = payload.get("person_name")
        if name:
            for worker in self.repository.list_entities(kind="worker"):
                if worker["data"].get("name") == name:
                    return worker
        return None

    def _find_refuge(self, payload):
        refuge = self.repository.get_entity(payload.get("refuge_id"))
        if refuge and refuge["kind"] == "refuge":
            return refuge
        return None

    def _find_ventilation(self, payload):
        vent = self.repository.get_entity(payload.get("ventilation_id"))
        if vent and vent["kind"] == "ventilation":
            return vent
        return None

    def _find_sensors(self, payload):
        sid = payload.get("sensor_id")
        if sid:
            sensor = self.repository.get_entity(sid)
            return [sensor] if sensor and sensor["kind"] == "sensor" else []
        loc = payload.get("location_code")
        if loc:
            return [s for s in self.repository.list_entities(kind="sensor") if s["data"].get("location_code") == loc]
        return []

    def _sensors_for_area(self, area):
        return [s for s in self.repository.list_entities(kind="sensor") if s["data"].get("location_code") == area]

    def _person_occupancy_state(self):
        """从已落地的 occupancy 记录中推出每个人的最新占用状态。

        返回 person_id -> (action, refuge_id, seq, record_entity_id)。
        只统计 status=applied 的记录，因此被标记冲突或拒绝的记录不计入。
        顺序按合并序号 _seq 判断先后，而非 recorded_at。
        """
        state = {}
        for record in self.repository.list_entities(kind="offline_record"):
            if record["status"] != "applied":
                continue
            payload = record["data"].get("payload", {})
            if payload.get("type") != "occupancy":
                continue
            pid = payload.get("person_id")
            if not pid:
                continue
            seq = int(record["data"].get("_seq", 0))
            current = state.get(pid)
            if current is None or seq > current[2]:
                state[pid] = (payload.get("action"), payload.get("refuge_id"), seq, record["id"])
        return state

    def _refuge_occupants(self, refuge_id):
        state = self._person_occupancy_state()
        return {
            pid
            for pid, (action, rid, _ts, _eid) in state.items()
            if action == "occupy" and rid == refuge_id
        }

    def _refresh_refuge_occupied(self, refuge_id):
        refuge = self.repository.get_entity(refuge_id)
        if not refuge:
            return
        data = dict(refuge["data"])
        data["occupied"] = len(self._refuge_occupants(refuge_id))
        self.repository.update_entity(refuge["id"], refuge["version"], refuge["status"], data)

    def _mark_conflict(self, entity, prior_record_id, reason):
        # 两版都保留：把先前落地的一版也标记为冲突，互不覆盖。
        if prior_record_id:
            prior = self.repository.get_entity(prior_record_id)
            if prior and prior["status"] == "applied":
                self._set_record_status(prior, "conflict", reason=reason, conflict_with=entity["id"])
        return self._set_record_status(entity, "conflict", reason=reason, conflict_with=prior_record_id)

    def _apply_occupancy(self, actor, entity, payload):
        person = self._find_person(payload)
        if not person:
            return self._set_record_status(entity, "rejected", reason="person not found")
        refuge = self._find_refuge(payload)
        if not refuge:
            return self._set_record_status(entity, "rejected", reason="refuge not found")
        action = payload["action"]
        state = self._person_occupancy_state()
        current = state.get(person["id"])

        # 同一个人的记录对不上：当前占用的硐室与本记录不一致，两版都标冲突。
        if action == "occupy":
            if current and current[0] == "occupy" and current[1] != refuge["id"]:
                return self._mark_conflict(entity, current[3], "person already occupies another refuge")
        elif action == "release":
            if current and current[0] == "occupy" and current[1] != refuge["id"]:
                return self._mark_conflict(entity, current[3], "person occupies another refuge")

        # 容量核对：超出核定容量的占用拒绝落地。
        if action == "occupy":
            capacity = float(refuge["data"].get("capacity", 0))
            occupants = self._refuge_occupants(refuge["id"])
            if person["id"] not in occupants and len(occupants) + 1 > capacity:
                self._refresh_refuge_occupied(refuge["id"])
                return self._set_record_status(entity, "rejected", reason="refuge capacity exceeded")

        result = self._set_record_status(entity, "applied")
        self._refresh_refuge_occupied(refuge["id"])
        return result

    def _apply_recovery_request(self, actor, entity, payload):
        vent = self._find_ventilation(payload)
        if not vent:
            return self._set_record_status(entity, "rejected", reason="ventilation not found")
        # 恢复申请只登记为待处理，不立即落地；等气体降到阈值以下并经现场确认。
        return self._set_record_status(entity, "pending")

    def _void_pending_recoveries(self, area):
        for record in self.repository.list_entities(kind="offline_record"):
            if record["status"] != "pending":
                continue
            payload = record["data"].get("payload", {})
            if payload.get("type") != "recovery":
                continue
            if area is not None:
                vent = self.repository.get_entity(payload.get("ventilation_id"))
                if not vent or vent["data"].get("area_code") != area:
                    continue
            self._set_record_status(record, "voided", reason="gas reading changed")

    def _apply_gas(self, actor, entity, payload):
        value = float(payload.get("value"))
        sensors = self._find_sensors(payload)
        areas = set()
        for sensor in sensors:
            data = dict(sensor["data"])
            data["gas_ppm"] = value
            threshold = float(data.get("threshold_ppm", 0))
            data["severity"] = (
                "alarm" if value >= threshold * 1.5
                else "warning" if value >= threshold
                else "normal"
            )
            self.repository.update_entity(sensor["id"], sensor["version"], sensor["status"], data)
            areas.add(sensor["data"].get("location_code"))
        # 气体读数一变，还没处理的恢复申请就作废，需现场确认后再落地。
        # 只作废同一区域内通风机的待处理申请；无法关联区域时不波及其他区域。
        for area in areas:
            self._void_pending_recoveries(area)
        return self._set_record_status(entity, "applied" if sensors else "merged")

    def _apply_recovery_confirm(self, actor, entity, payload):
        vent = self._find_ventilation(payload)
        if not vent:
            return self._set_record_status(entity, "rejected", reason="ventilation not found")
        request = None
        for record in self.repository.list_entities(kind="offline_record"):
            rpayload = record["data"].get("payload", {})
            if (
                rpayload.get("type") == "recovery"
                and rpayload.get("ventilation_id") == vent["id"]
                and record["status"] in ("pending", "voided")
            ):
                if request is None or int(record["data"].get("_seq", 0)) > int(request["data"].get("_seq", 0)):
                    request = record
        if not request:
            return self._set_record_status(entity, "rejected", reason="no pending recovery request")

        # 气体还没降到阈值以下就不能恢复运行。
        for sensor in self._sensors_for_area(vent["data"].get("area_code")):
            if float(sensor["data"].get("gas_ppm", 0)) >= float(sensor["data"].get("threshold_ppm", 1)):
                return self._set_record_status(entity, "rejected", reason="gas above threshold")

        if vent["status"] in ("stopped", "degraded"):
            try:
                next_status, patch = self.rules.validate_transition(
                    actor, vent, "restore", {"tested_at": utcnow()}, self._lookup
                )
            except PermissionDenied:
                return self._set_record_status(entity, "rejected", reason="restore not permitted for role")
            merged = dict(vent["data"])
            merged.update(patch)
            self.repository.update_entity(vent["id"], vent["version"], next_status, merged)
            self.audit.record(vent["id"], actor, "restore", vent["status"], next_status, {"source": "offline_merge"})
        self._set_record_status(request, "applied")
        return self._set_record_status(entity, "applied")

    def _apply_conflict_resolve(self, actor, entity, payload):
        person = self._find_person(payload)
        if not person:
            return self._set_record_status(entity, "rejected", reason="person not found")
        resolved = 0
        for record in self.repository.list_entities(kind="offline_record"):
            rpayload = record["data"].get("payload", {})
            if (
                rpayload.get("type") == "occupancy"
                and rpayload.get("person_id") == person["id"]
                and record["status"] == "conflict"
            ):
                self._set_record_status(record, "resolved")
                resolved += 1
        return self._set_record_status(entity, "applied" if resolved else "merged")

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

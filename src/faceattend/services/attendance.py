from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from ..core.config import Settings
from ..infrastructure.faiss_index import SearchCandidate
from ..infrastructure.mongo_repository import MongoRepository


@dataclass
class CandidateState:
    employee_id: str
    hits: int
    first_seen: datetime
    last_seen: datetime


class TemporalConfirmation:
    def __init__(self, window_seconds: float):
        self._window_seconds = window_seconds
        self._states: dict[str, CandidateState] = {}
        self._lock = threading.Lock()

    def observe(self, source_id: str, employee_id: str, now: datetime) -> int:
        with self._lock:
            state = self._states.get(source_id)
            expired = state and (now - state.last_seen).total_seconds() > self._window_seconds
            if state is None or expired or state.employee_id != employee_id:
                self._states[source_id] = CandidateState(employee_id, 1, now, now)
                return 1
            state.hits += 1
            state.last_seen = now
            return state.hits

    def reset(self, source_id: str) -> None:
        with self._lock:
            self._states.pop(source_id, None)


class AttendanceService:
    def __init__(self, repository: MongoRepository, settings: Settings):
        self._repository = repository
        self._settings = settings

    def record(
        self,
        *,
        candidate: SearchCandidate,
        camera_id: str,
        direction: str,
        match_score: float,
        quality_score: float,
    ) -> tuple[dict, bool]:
        now = datetime.now(timezone.utc)
        last_event = self._repository.find_last_event(candidate.employee_id, camera_id)
        event_type = self._choose_direction(direction, last_event)
        bucket = math.floor(now.timestamp() / self._settings.attendance_cooldown_seconds)
        idempotency_key = f"{candidate.employee_id}:{camera_id}:{bucket}"
        document = {
            "employee_id": self._repository._object_id(candidate.employee_id),
            "employee_code": candidate.employee_code,
            "full_name": candidate.full_name,
            "camera_id": camera_id,
            "event_type": event_type,
            "occurred_at": now,
            "match_score": match_score,
            "quality_score": quality_score,
            "model_version": self._settings.biometric_model_version,
            "idempotency_key": idempotency_key,
        }
        event, created = self._repository.create_attendance_event(document)
        return self._repository.attendance_read(event), created

    @staticmethod
    def _choose_direction(requested: str, last_event: dict | None) -> str:
        if requested in {"check_in", "check_out"}:
            return requested
        if last_event and last_event["event_type"] == "check_in":
            return "check_out"
        return "check_in"

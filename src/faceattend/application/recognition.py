from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
from fastapi import HTTPException, status

from .biometrics import (
    extract_faces_or_error,
    liveness_read,
    quality_message,
    quality_read,
)
from .container import AppContainer
from ..domain.schemas import RecognitionRead
from ..infrastructure.faiss_index import SearchCandidate
from ..infrastructure.mongo_repository import utcnow
from ..services.face_engine import DetectedFace
from ..services.liveness import LivenessResult
from ..services.quality import QualityAssessment


@dataclass
class MatchDecision:
    candidate: SearchCandidate | None
    top_candidate: SearchCandidate | None
    reason: str | None


def process_frame(
    container: AppContainer,
    image_bytes: bytes,
    client_id: str,
    camera_id: str,
    direction: str,
    dry_run: bool = False,
) -> RecognitionRead:
    started_at = time.perf_counter()
    if direction not in {"auto", "check_in", "check_out"}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid attendance direction")
    try:
        faces = extract_faces_or_error(image_bytes, container)
    except HTTPException:
        container.metrics.record("recognition", "error", _latency_ms(started_at))
        raise
    if not faces:
        if not dry_run:
            container.confirmation.reset(client_id)
        return _response(
            container,
            started_at,
            RecognitionRead(status="no_face", message="Không phát hiện khuôn mặt"),
        )
    if len(faces) > 1:
        if not dry_run:
            container.confirmation.reset(client_id)
        return _response(
            container,
            started_at,
            RecognitionRead(status="multiple_faces", message="Chỉ để một khuôn mặt trong khung hình"),
        )
    face = faces[0]
    quality = container.quality.assess(face, "recognition")
    if not quality.passed:
        if not dry_run:
            container.confirmation.reset(client_id)
        return _response(
            container,
            started_at,
            RecognitionRead(
                status="poor_quality",
                message=quality_message(quality),
                quality_score=quality.score,
                quality=quality_read(quality),
            ),
        )
    liveness = container.liveness.evaluate(face)
    if liveness.status == "unavailable":
        return _response(
            container,
            started_at,
            RecognitionRead(
                status="liveness_unavailable",
                message=liveness.message,
                quality_score=quality.score,
                quality=quality_read(quality),
                liveness=liveness_read(liveness),
            ),
        )
    if liveness.status == "failed":
        if not dry_run:
            container.confirmation.reset(client_id)
        return _response(
            container,
            started_at,
            RecognitionRead(
                status="liveness_failed",
                message=liveness.message,
                quality_score=quality.score,
                quality=quality_read(quality),
                liveness=liveness_read(liveness),
            ),
        )
    decision = _match_decision(container, face.embedding)
    if decision.candidate is None:
        if not dry_run:
            container.confirmation.reset(client_id)
            _create_unknown_review(
                container=container,
                client_id=client_id,
                camera_id=camera_id,
                reason=decision.reason or "UNKNOWN",
                face=face,
                quality=quality,
                liveness=liveness,
                decision=decision,
            )
        return _response(
            container,
            started_at,
            RecognitionRead(
                status="unknown",
                message=decision.reason or "Không xác định được nhân viên",
                quality_score=quality.score,
                match_score=decision.top_candidate.score if decision.top_candidate else None,
                quality=quality_read(quality),
                liveness=liveness_read(liveness),
            ),
        )
    candidate = decision.candidate
    if dry_run:
        return _response(
            container,
            started_at,
            RecognitionRead(
                status="pending",
                message="Benchmark recognition completed",
                employee_code=candidate.employee_code,
                full_name=candidate.full_name,
                match_score=candidate.score,
                quality_score=quality.score,
                quality=quality_read(quality),
                liveness=liveness_read(liveness),
            ),
        )
    hits = container.confirmation.observe(client_id, candidate.employee_id, datetime.now(timezone.utc))
    if hits < container.settings.confirmation_frames:
        return _response(
            container,
            started_at,
            RecognitionRead(
                status="pending",
                message="Đã thấy danh tính, đang lấy thêm frame xác nhận",
                employee_code=candidate.employee_code,
                full_name=candidate.full_name,
                match_score=candidate.score,
                quality_score=quality.score,
                quality=quality_read(quality),
                liveness=liveness_read(liveness),
                confirmation_hits=hits,
            ),
        )
    event, created = container.attendance.record(
        candidate=candidate,
        camera_id=camera_id,
        direction=direction,
        match_score=candidate.score,
        quality_score=quality.score,
    )
    return _response(
        container,
        started_at,
        RecognitionRead(
            status="confirmed" if created else "cooldown",
            message="Đã ghi nhận chấm công" if created else "Nhân viên đã nằm trong cooldown",
            employee_code=candidate.employee_code,
            full_name=candidate.full_name,
            match_score=candidate.score,
            quality_score=quality.score,
            quality=quality_read(quality),
            liveness=liveness_read(liveness),
            confirmation_hits=hits,
            event=event,
        ),
    )


def _match_decision(container: AppContainer, embedding) -> MatchDecision:
    matches = container.index.search(embedding, top_k=32)
    by_employee: dict[str, SearchCandidate] = {}
    for match in matches:
        current = by_employee.get(match.employee_id)
        if current is None or match.score > current.score:
            by_employee[match.employee_id] = match
    ranked = sorted(by_employee.values(), key=lambda item: item.score, reverse=True)
    if not ranked:
        return MatchDecision(None, None, "Không có template nào đã enrollment")
    top_candidate = ranked[0]
    if top_candidate.score < container.settings.match_threshold:
        return MatchDecision(None, top_candidate, "Điểm nhận diện chưa đạt ngưỡng")
    if (
        len(ranked) > 1
        and top_candidate.score - ranked[1].score < container.settings.match_margin
    ):
        return MatchDecision(None, top_candidate, "Kết quả gần hai nhân sự khác nhau")
    return MatchDecision(top_candidate, top_candidate, None)


def _create_unknown_review(
    *,
    container: AppContainer,
    client_id: str,
    camera_id: str,
    reason: str,
    face: DetectedFace,
    quality: QualityAssessment,
    liveness: LivenessResult,
    decision: MatchDecision,
) -> None:
    bucket = math.floor(time.time() / container.settings.review_cooldown_seconds)
    top_candidate = decision.top_candidate
    container.repository.create_review(
        {
            "status": "pending",
            "reason": reason,
            "camera_id": camera_id,
            "client_id": client_id,
            "quality_score": quality.score,
            "quality_reasons": quality.reasons,
            "match_score": top_candidate.score if top_candidate else None,
            "candidate_employee_code": top_candidate.employee_code if top_candidate else None,
            "liveness_status": liveness.status,
            "liveness_score": liveness.score,
            "evidence_encrypted": _encode_review_evidence(face, container),
            "dedup_key": f"{client_id}:{camera_id}:{reason}:{bucket}",
            "created_at": utcnow(),
        }
    )


def _encode_review_evidence(face: DetectedFace, container: AppContainer) -> bytes | None:
    if not container.settings.review_evidence_enabled:
        return None
    max_pixels = container.settings.review_evidence_max_pixels
    height, width = face.crop.shape[:2]
    scale = min(1.0, max_pixels / max(height, width))
    thumbnail = face.crop
    if scale < 1.0:
        thumbnail = cv2.resize(face.crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    encoded, image_bytes = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not encoded:
        return None
    return container.cipher.encrypt_bytes(image_bytes.tobytes())


def _response(
    container: AppContainer, started_at: float, response: RecognitionRead
) -> RecognitionRead:
    container.metrics.record("recognition", response.status, _latency_ms(started_at))
    return response


def _latency_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000

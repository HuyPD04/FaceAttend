from __future__ import annotations

import time

from fastapi import HTTPException, status

from .biometrics import liveness_read, quality_read, single_face_or_error
from .container import AppContainer
from ..infrastructure.mongo_repository import NotFoundError
from ..services.face_engine import DetectedFace
from ..services.quality import QualityAssessment


def start_session(container: AppContainer, employee_id: str) -> dict:
    try:
        return container.repository.start_enrollment_session(
            employee_id,
            container.settings.enrollment_target_samples,
            container.settings.enrollment_min_samples,
        )
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def complete_session(container: AppContainer, session_id: str) -> dict:
    try:
        return container.repository.complete_enrollment_session(session_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def capture_session_frame(container: AppContainer, session_id: str, image_bytes: bytes) -> dict:
    started_at = time.perf_counter()
    try:
        session = container.repository.get_enrollment_session(session_id)
        if session["status"] != "active":
            raise HTTPException(status.HTTP_409_CONFLICT, "Enrollment session đã hoàn thành")
        face, quality, liveness = single_face_or_error(image_bytes, container, "enrollment")
        created = create_template(
            container=container,
            employee_id=str(session["employee_id"]),
            face=face,
            quality=quality,
            require_distinct_sample=True,
        )
        try:
            updated_session = container.repository.record_enrollment_sample(
                session_id=session_id,
                template_id=created["template_id"],
                quality_score=quality.score,
                pose_bucket=quality.pose_bucket,
                yaw_degrees=quality.yaw_degrees,
                roll_degrees=quality.roll_degrees,
            )
        except (NotFoundError, ValueError) as exc:
            container.repository.delete_template(created["faiss_id"])
            container.index.remove([created["faiss_id"]])
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        container.metrics.record("enrollment", "accepted", _latency_ms(started_at))
        return {
            "template": created,
            "session": updated_session,
            "quality": quality_read(quality),
            "liveness": liveness_read(liveness),
        }
    except HTTPException:
        container.metrics.record("enrollment", "rejected", _latency_ms(started_at))
        raise


def enroll_single_template(container: AppContainer, employee_id: str, image_bytes: bytes) -> dict:
    started_at = time.perf_counter()
    try:
        face, quality, _ = single_face_or_error(image_bytes, container, "enrollment")
        created = create_template(
            container=container,
            employee_id=employee_id,
            face=face,
            quality=quality,
            require_distinct_sample=False,
        )
        container.metrics.record("enrollment", "accepted", _latency_ms(started_at))
        return created
    except HTTPException:
        container.metrics.record("enrollment", "rejected", _latency_ms(started_at))
        raise


def deactivate_employee(container: AppContainer, employee_id: str) -> None:
    try:
        faiss_ids = container.repository.deactivate_employee(employee_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    container.index.remove(faiss_ids)


def create_template(
    *,
    container: AppContainer,
    employee_id: str,
    face: DetectedFace,
    quality: QualityAssessment,
    require_distinct_sample: bool,
) -> dict:
    try:
        employee = container.repository.get_employee(employee_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    for candidate in container.index.search(face.embedding, top_k=32):
        if (
            candidate.employee_id != employee_id
            and candidate.score >= container.settings.duplicate_threshold
        ):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Khuôn mặt có thể đã thuộc nhân viên {candidate.employee_code}",
            )
    same_similarity = _same_employee_similarity(container, employee_id, face.embedding)
    if (
        require_distinct_sample
        and same_similarity is not None
        and same_similarity >= container.settings.enrollment_duplicate_similarity
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Mẫu quá giống ảnh đã có; hãy đổi nhẹ góc nhìn hoặc khoảng cách",
        )
    faiss_id = container.repository.reserve_faiss_id()
    created = container.repository.create_template(
        employee_id=employee_id,
        faiss_id=faiss_id,
        embedding_encrypted=container.cipher.encrypt(face.embedding),
        quality_score=quality.score,
        model_version=container.settings.biometric_model_version,
        embedding_model_id=container.settings.recognizer_model_id,
        detector_model_id=container.settings.detector_model_id,
        preprocess_version=container.settings.recognizer_preprocess_version,
        embedding_dimension=container.settings.embedding_dimension,
    )
    try:
        container.index.add(
            faiss_id=faiss_id,
            embedding=face.embedding,
            employee_id=employee_id,
            employee_code=employee["employee_code"],
            full_name=employee["full_name"],
        )
    except Exception as exc:
        container.repository.delete_template(faiss_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Không thể cập nhật FAISS index"
        ) from exc
    return created


def _same_employee_similarity(container: AppContainer, employee_id: str, embedding) -> float | None:
    scores = [
        candidate.score
        for candidate in container.index.search(embedding, top_k=32)
        if candidate.employee_id == employee_id
    ]
    return max(scores) if scores else None


def _latency_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000

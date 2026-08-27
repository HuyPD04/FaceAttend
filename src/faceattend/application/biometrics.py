from __future__ import annotations

from typing import Literal

from fastapi import HTTPException, status

from .container import AppContainer
from ..services.face_engine import DetectedFace, InferenceUnavailableError
from ..services.liveness import LivenessResult
from ..services.quality import QualityAssessment


Purpose = Literal["enrollment", "recognition"]


def quality_read(assessment: QualityAssessment) -> dict:
    return {
        "passed": assessment.passed,
        "score": assessment.score,
        "reasons": assessment.reasons,
        "pose_bucket": assessment.pose_bucket,
        "yaw_degrees": assessment.yaw_degrees,
        "roll_degrees": assessment.roll_degrees,
        "sharpness": assessment.sharpness,
        "brightness": assessment.brightness,
    }


def liveness_read(result: LivenessResult) -> dict:
    return {
        "status": result.status,
        "provider": result.provider,
        "score": result.score,
        "message": result.message,
    }


def quality_message(assessment: QualityAssessment) -> str:
    messages = {
        "FACE_TOO_SMALL": "Đưa khuôn mặt lại gần camera hơn",
        "DETECTION_LOW": "Camera chưa nhìn rõ khuôn mặt",
        "IMAGE_BLURRY": "Giữ yên khuôn mặt để ảnh bớt mờ",
        "TOO_DARK": "Tăng ánh sáng ở phía trước khuôn mặt",
        "TOO_BRIGHT": "Giảm ánh sáng quá gắt trước khuôn mặt",
        "FACE_TURNED_TOO_FAR": "Quay mặt gần chính diện hơn",
        "FACE_TILTED": "Giữ đầu thẳng hơn",
    }
    return ". ".join(messages[reason] for reason in assessment.reasons)


def extract_faces_or_error(image_bytes: bytes, container: AppContainer) -> list[DetectedFace]:
    try:
        return container.engine.extract(image_bytes)
    except InferenceUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def single_face_or_error(
    image_bytes: bytes, container: AppContainer, purpose: Purpose
) -> tuple[DetectedFace, QualityAssessment, LivenessResult]:
    faces = extract_faces_or_error(image_bytes, container)
    if not faces:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Không phát hiện được khuôn mặt")
    if len(faces) > 1:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Chỉ để một khuôn mặt trong khung hình")
    face = faces[0]
    assessment = container.quality.assess(face, purpose)
    if not assessment.passed:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, quality_message(assessment))
    liveness = container.liveness.evaluate(face)
    if liveness.status == "unavailable":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, liveness.message)
    if liveness.status == "failed":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, liveness.message)
    return face, assessment, liveness

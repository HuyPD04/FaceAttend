from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core.config import Settings
from .face_engine import DetectedFace


QualityPurpose = Literal["recognition", "enrollment"]


@dataclass(frozen=True)
class QualityAssessment:
    passed: bool
    score: float
    reasons: list[str]
    pose_bucket: str
    yaw_degrees: float
    roll_degrees: float
    sharpness: float
    brightness: float


class FaceQualityGate:
    def __init__(self, settings: Settings):
        self._settings = settings

    def assess(self, face: DetectedFace, purpose: QualityPurpose) -> QualityAssessment:
        reasons: list[str] = []
        if face.face_pixels < self._settings.min_face_pixels:
            reasons.append("FACE_TOO_SMALL")
        if face.detection_score < self._settings.min_detection_score:
            reasons.append("DETECTION_LOW")
        if face.sharpness < self._settings.min_sharpness:
            reasons.append("IMAGE_BLURRY")
        if face.brightness < self._settings.min_brightness:
            reasons.append("TOO_DARK")
        if face.brightness > self._settings.max_brightness:
            reasons.append("TOO_BRIGHT")
        if purpose == "enrollment":
            if abs(face.yaw_degrees) > self._settings.max_enrollment_yaw_degrees:
                reasons.append("FACE_TURNED_TOO_FAR")
            if abs(face.roll_degrees) > self._settings.max_enrollment_roll_degrees:
                reasons.append("FACE_TILTED")
        score = min(
            face.detection_score / 0.95,
            face.face_pixels / (self._settings.min_face_pixels * 1.5),
            face.sharpness / (self._settings.min_sharpness * 2.0),
            self._brightness_score(face.brightness),
            self._pose_score(face, purpose),
        )
        return QualityAssessment(
            passed=not reasons,
            score=round(float(max(0.0, min(score, 1.0))), 4),
            reasons=reasons,
            pose_bucket=self._pose_bucket(face.yaw_degrees),
            yaw_degrees=round(face.yaw_degrees, 2),
            roll_degrees=round(face.roll_degrees, 2),
            sharpness=round(face.sharpness, 2),
            brightness=round(face.brightness, 2),
        )

    def _brightness_score(self, brightness: float) -> float:
        low = self._settings.min_brightness
        high = self._settings.max_brightness
        if low <= brightness <= high:
            return 1.0
        if brightness < low:
            return brightness / max(low, 1.0)
        return max(0.0, 1.0 - (brightness - high) / max(255.0 - high, 1.0))

    def _pose_score(self, face: DetectedFace, purpose: QualityPurpose) -> float:
        if purpose == "recognition":
            return 1.0
        yaw = abs(face.yaw_degrees) / max(self._settings.max_enrollment_yaw_degrees, 1.0)
        roll = abs(face.roll_degrees) / max(self._settings.max_enrollment_roll_degrees, 1.0)
        return max(0.0, 1.0 - max(yaw, roll))

    @staticmethod
    def _pose_bucket(yaw_degrees: float) -> str:
        if yaw_degrees <= -8.0:
            return "left"
        if yaw_degrees >= 8.0:
            return "right"
        return "front"

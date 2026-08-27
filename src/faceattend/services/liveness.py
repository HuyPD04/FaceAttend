from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..core.config import Settings
from .face_engine import DetectedFace


@dataclass(frozen=True)
class LivenessResult:
    status: str
    provider: str
    score: float | None
    message: str


class DisabledLivenessProvider:
    def evaluate(self, face: DetectedFace) -> LivenessResult:
        return LivenessResult("disabled", "disabled", None, "Liveness is disabled")


class UnavailableLivenessProvider:
    def __init__(self, message: str):
        self._message = message

    def evaluate(self, face: DetectedFace) -> LivenessResult:
        return LivenessResult("unavailable", "unavailable", None, self._message)


class OnnxLivenessProvider:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._session: ort.InferenceSession | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def evaluate(self, face: DetectedFace) -> LivenessResult:
        session = self._ensure_session()
        if session is None:
            return LivenessResult("unavailable", "onnx", None, self._error or "Model unavailable")
        size = self._settings.liveness_input_size
        image = cv2.resize(face.crop, (size, size), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = (image.astype(np.float32) - 127.5) / 128.0
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...])
        try:
            logits = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0]).reshape(-1)
        except Exception as exc:
            return LivenessResult("unavailable", "onnx", None, f"Liveness inference failed: {exc}")
        probabilities = self._softmax(logits)
        index = self._settings.liveness_real_class_index
        if index < 0 or index >= len(probabilities):
            return LivenessResult("unavailable", "onnx", None, "Invalid real-class index")
        score = float(probabilities[index])
        passed = score >= self._settings.liveness_threshold
        return LivenessResult(
            "passed" if passed else "failed",
            "onnx",
            round(score, 4),
            "Liveness passed" if passed else "Liveness rejected",
        )

    def _ensure_session(self) -> ort.InferenceSession | None:
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            if self._error is not None:
                return None
            path = Path(self._settings.liveness_model_path)
            if not path.is_file():
                self._error = f"Liveness model not found: {path}"
                return None
            try:
                self._session = ort.InferenceSession(
                    str(path), providers=self._settings.provider_list
                )
                self._error = None
                return self._session
            except Exception as exc:
                self._error = f"Unable to load liveness model: {exc}"
                return None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)


class LivenessService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._provider = self._build_provider()

    def evaluate(self, face: DetectedFace) -> LivenessResult:
        return self._provider.evaluate(face)

    def _build_provider(self):
        if self._settings.liveness_mode == "disabled":
            return DisabledLivenessProvider()
        if self._settings.liveness_mode == "onnx":
            return OnnxLivenessProvider(self._settings)
        return UnavailableLivenessProvider(
            f"Unsupported liveness mode: {self._settings.liveness_mode}"
        )

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "faceattend"
    data_dir: Path = Path("data")
    faiss_index_path: Path = Path("data/faiss/templates.index")
    fernet_key_file: Path | None = None

    encryption_key: str = Field(default="", validation_alias="FACEATTEND_ENCRYPTION_KEY")
    onnx_providers: str = "CPUExecutionProvider"
    detector_model_path: Path = Path("data/models/models/buffalo_l/det_10g.onnx")
    detector_model_id: str = "scrfd-10g-bnkps-v1"
    detector_input_size: int = 640
    detector_threshold: float = 0.50
    detector_nms_threshold: float = 0.40
    recognizer_model_path: Path = Path("data/models/models/buffalo_l/w600k_r50.onnx")
    recognizer_model_id: str = "arcface-w600k-r50-v1"
    recognizer_input_size: int = 112
    recognizer_preprocess_version: str = "arcface-112x112-5pt-v1"
    embedding_dimension: int = 512

    match_threshold: float = 0.48
    match_margin: float = 0.06
    duplicate_threshold: float = 0.62
    min_face_pixels: int = 120
    min_detection_score: float = 0.80
    min_sharpness: float = 60.0
    min_brightness: float = 45.0
    max_brightness: float = 215.0
    max_enrollment_yaw_degrees: float = 28.0
    max_enrollment_roll_degrees: float = 15.0
    confirmation_frames: int = 3
    confirmation_window_seconds: float = 2.0
    attendance_cooldown_seconds: int = 600
    enrollment_target_samples: int = 5
    enrollment_min_samples: int = 3
    enrollment_duplicate_similarity: float = 0.997
    liveness_mode: str = "disabled"
    liveness_model_path: Path = Path("data/models/liveness.onnx")
    liveness_threshold: float = 0.90
    liveness_input_size: int = 80
    liveness_real_class_index: int = 1
    review_evidence_enabled: bool = True
    review_evidence_max_pixels: int = 240
    review_cooldown_seconds: int = 60

    @property
    def fernet_key(self) -> bytes:
        configured_key = self.encryption_key.strip()
        if configured_key:
            return configured_key.encode("utf-8")
        return self._load_or_create_fernet_key()

    @property
    def resolved_fernet_key_file(self) -> Path:
        return self.fernet_key_file or self.data_dir / "fernet.key"

    def _load_or_create_fernet_key(self) -> bytes:
        key_file = self.resolved_fernet_key_file
        key_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            return key_file.read_bytes().strip()
        except FileNotFoundError:
            generated_key = Fernet.generate_key()
            try:
                descriptor = os.open(
                    key_file,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                return key_file.read_bytes().strip()
            with os.fdopen(descriptor, "wb") as key_handle:
                key_handle.write(generated_key)
            return generated_key

    @property
    def provider_list(self) -> list[str]:
        return [item.strip() for item in self.onnx_providers.split(",") if item.strip()]

    @property
    def biometric_model_version(self) -> str:
        return f"{self.recognizer_model_id}:{self.recognizer_preprocess_version}"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

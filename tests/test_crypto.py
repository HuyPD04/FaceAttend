import numpy as np
from cryptography.fernet import Fernet

from faceattend.services.crypto import EmbeddingCipher
from faceattend.core.config import Settings
from faceattend.services.face_engine import DetectedFace, FaceEngine
from faceattend.services.quality import FaceQualityGate
from faceattend.services.metrics import MetricsRegistry
from scripts.calibrate_thresholds import calibrate
from scripts.benchmark_latency import percentile


def test_encrypted_embedding_round_trip() -> None:
    key = Fernet.generate_key()
    cipher = EmbeddingCipher(key, 4)
    original = np.asarray([1, 2, 3, 4], dtype=np.float32)
    restored = cipher.decrypt(cipher.encrypt(original))
    assert np.isclose(np.linalg.norm(restored), 1.0)
    assert np.allclose(restored, original / np.linalg.norm(original))


def test_missing_configured_key_is_generated_once(tmp_path) -> None:
    key_file = tmp_path / "fernet.key"
    first = Settings(encryption_key="", fernet_key_file=key_file)
    second = Settings(encryption_key="", fernet_key_file=key_file)
    assert first.fernet_key == second.fernet_key
    assert len(first.fernet_key) == 44


def test_quality_gate_reports_blurry_frame() -> None:
    settings = Settings(_env_file=None, min_sharpness=60.0)
    face = DetectedFace(
        embedding=np.ones(512, dtype=np.float32),
        detection_score=0.99,
        quality_score=0.9,
        face_pixels=180,
        crop=np.zeros((180, 180, 3), dtype=np.uint8),
        sharpness=10.0,
        brightness=128.0,
        yaw_degrees=0.0,
        roll_degrees=0.0,
    )
    assessment = FaceQualityGate(settings).assess(face, "enrollment")
    assert not assessment.passed
    assert "IMAGE_BLURRY" in assessment.reasons


def test_calibration_respects_target_far() -> None:
    result = calibrate([0.92, 0.88, 0.81], [0.12, 0.24, 0.76], 0.0)
    assert result["measured_far"] == 0.0
    assert result["match_threshold"] > 0.76


def test_scrfd_anchor_layout_uses_two_anchors_per_cell() -> None:
    centers = FaceEngine._anchor_centers(640, 8, 12800)
    assert centers.shape == (12800, 2)
    assert np.array_equal(centers[:3], np.asarray([[0, 0], [0, 0], [8, 0]], dtype=np.float32))


def test_default_models_are_explicit_scrfd_and_arcface() -> None:
    settings = Settings(_env_file=None)
    assert settings.detector_model_id.startswith("scrfd-")
    assert settings.recognizer_model_id.startswith("arcface-")
    assert settings.biometric_model_version.startswith(settings.recognizer_model_id)


def test_metrics_reports_latency_percentiles() -> None:
    registry = MetricsRegistry()
    for latency in [10, 20, 30, 40, 50]:
        registry.record("recognition", "pending", latency)
    metrics = registry.snapshot()["operations"]["recognition"]
    assert metrics["p50_latency_ms"] == 30.0
    assert metrics["p95_latency_ms"] == 48.0
    assert metrics["p99_latency_ms"] == 49.6


def test_benchmark_percentile_uses_linear_interpolation() -> None:
    assert percentile([10, 20, 30, 40, 50], 95) == 48.0

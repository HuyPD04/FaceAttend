from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


AttendanceDirection = Literal["auto", "check_in", "check_out"]


class EmployeeCreate(BaseModel):
    employee_code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    full_name: str = Field(min_length=1, max_length=160)
    site_id: str = Field(default="default", min_length=1, max_length=64)


class EmployeeRead(BaseModel):
    id: str
    employee_code: str
    full_name: str
    site_id: str
    status: Literal["active", "inactive"]
    template_count: int
    created_at: datetime


class AttendanceRead(BaseModel):
    id: str
    employee_id: str
    employee_code: str
    full_name: str
    camera_id: str
    event_type: Literal["check_in", "check_out"]
    occurred_at: datetime
    match_score: float
    quality_score: float
    model_version: str


class EnrollmentRead(BaseModel):
    template_id: str
    employee_id: str
    quality_score: float
    model_version: str


class QualityRead(BaseModel):
    passed: bool
    score: float
    reasons: list[str]
    pose_bucket: str
    yaw_degrees: float
    roll_degrees: float
    sharpness: float
    brightness: float


class LivenessRead(BaseModel):
    status: str
    provider: str
    score: float | None = None
    message: str


class EnrollmentSessionRead(BaseModel):
    id: str
    employee_id: str
    status: Literal["active", "completed"]
    target_samples: int
    min_samples: int
    sample_count: int
    pose_counts: dict[str, int]
    next_pose_hint: str
    created_at: datetime
    completed_at: datetime | None = None


class EnrollmentFrameRead(BaseModel):
    template: EnrollmentRead
    session: EnrollmentSessionRead
    quality: QualityRead
    liveness: LivenessRead


class ReviewRead(BaseModel):
    id: str
    status: Literal["pending", "dismissed"]
    reason: str
    camera_id: str
    quality_score: float
    match_score: float | None = None
    candidate_employee_code: str | None = None
    evidence_available: bool
    created_at: datetime


class RecognitionRead(BaseModel):
    status: Literal[
        "pending",
        "confirmed",
        "cooldown",
        "unknown",
        "no_face",
        "multiple_faces",
        "poor_quality",
        "liveness_failed",
        "liveness_unavailable",
    ]
    message: str
    employee_code: str | None = None
    full_name: str | None = None
    match_score: float | None = None
    quality_score: float | None = None
    quality: QualityRead | None = None
    liveness: LivenessRead | None = None
    confirmation_hits: int | None = None
    event: AttendanceRead | None = None


class HealthRead(BaseModel):
    status: Literal["ok", "degraded"]
    mongo: bool
    faiss_vectors: int
    model_loaded: bool
    model_error: str | None = None


class MetricsRead(BaseModel):
    started_at: datetime
    operations: dict[str, dict]

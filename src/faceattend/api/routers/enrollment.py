from fastapi import APIRouter, Depends, File, UploadFile, status

from ..dependencies import get_container
from ...application.container import AppContainer
from ...application.enrollment import (
    capture_session_frame,
    complete_session,
    enroll_single_template,
    start_session,
)
from ...domain.schemas import EnrollmentFrameRead, EnrollmentRead, EnrollmentSessionRead


router = APIRouter(tags=["enrollment"])


@router.post(
    "/employees/{employee_id}/enrollment-sessions",
    response_model=EnrollmentSessionRead,
    status_code=status.HTTP_201_CREATED,
)
def start_enrollment_session(
    employee_id: str, container: AppContainer = Depends(get_container)
) -> dict:
    return start_session(container, employee_id)


@router.post("/enrollment-sessions/{session_id}/frames", response_model=EnrollmentFrameRead)
def capture_enrollment_frame(
    session_id: str,
    image: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
) -> dict:
    return capture_session_frame(container, session_id, image.file.read())


@router.post("/enrollment-sessions/{session_id}/complete", response_model=EnrollmentSessionRead)
def complete_enrollment_session(
    session_id: str, container: AppContainer = Depends(get_container)
) -> dict:
    return complete_session(container, session_id)


@router.post("/employees/{employee_id}/templates", response_model=EnrollmentRead)
def enroll_template(
    employee_id: str,
    image: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
) -> dict:
    return enroll_single_template(container, employee_id, image.file.read())

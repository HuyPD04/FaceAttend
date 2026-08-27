from fastapi import APIRouter, Depends

from ..dependencies import get_container
from ...application.container import AppContainer
from ...domain.schemas import AttendanceRead


router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.get("", response_model=list[AttendanceRead])
def list_attendance(limit: int = 50, container: AppContainer = Depends(get_container)) -> list[dict]:
    return container.repository.list_attendance(limit=min(max(limit, 1), 200))

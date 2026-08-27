from fastapi import APIRouter, Depends, HTTPException, Response, status

from ..dependencies import get_container
from ...application.container import AppContainer
from ...application.enrollment import deactivate_employee
from ...domain.schemas import EmployeeCreate, EmployeeRead
from ...infrastructure.mongo_repository import DuplicateEmployeeError


router = APIRouter(prefix="/employees", tags=["employees"])


@router.get("", response_model=list[EmployeeRead])
def list_employees(container: AppContainer = Depends(get_container)) -> list[dict]:
    return container.repository.list_employees()


@router.post("", response_model=EmployeeRead, status_code=status.HTTP_201_CREATED)
def create_employee(
    payload: EmployeeCreate, container: AppContainer = Depends(get_container)
) -> dict:
    try:
        return container.repository.create_employee(
            payload.employee_code, payload.full_name, payload.site_id
        )
    except DuplicateEmployeeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def deactivate(employee_id: str, container: AppContainer = Depends(get_container)) -> Response:
    deactivate_employee(container, employee_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

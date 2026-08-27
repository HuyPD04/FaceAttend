from fastapi import APIRouter, Depends, HTTPException, Response, status

from ..dependencies import get_container
from ...application.container import AppContainer
from ...domain.schemas import ReviewRead
from ...infrastructure.mongo_repository import NotFoundError


router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("", response_model=list[ReviewRead])
def list_reviews(limit: int = 50, container: AppContainer = Depends(get_container)) -> list[dict]:
    return container.repository.list_reviews(limit=min(max(limit, 1), 200))


@router.post("/{review_id}/dismiss", response_model=ReviewRead)
def dismiss_review(review_id: str, container: AppContainer = Depends(get_container)) -> dict:
    try:
        return container.repository.dismiss_review(review_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/{review_id}/evidence", response_class=Response)
def review_evidence(review_id: str, container: AppContainer = Depends(get_container)) -> Response:
    try:
        encrypted = container.repository.get_review_evidence(review_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    if encrypted is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review không có evidence image")
    try:
        image_bytes = container.cipher.decrypt_bytes(encrypted)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Không thể giải mã evidence"
        ) from exc
    return Response(content=image_bytes, media_type="image/jpeg")

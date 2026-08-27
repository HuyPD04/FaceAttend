from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_container
from ...application.container import AppContainer
from ...domain.schemas import HealthRead, MetricsRead
from ...services.face_engine import InferenceUnavailableError


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthRead)
def health(container: AppContainer = Depends(get_container)) -> HealthRead:
    mongo_ok = container.repository.ping()
    return HealthRead(
        status="ok" if mongo_ok else "degraded",
        mongo=mongo_ok,
        faiss_vectors=container.index.size,
        model_loaded=container.engine.loaded,
        model_error=container.engine.load_error,
    )


@router.get("/metrics", response_model=MetricsRead)
def metrics(container: AppContainer = Depends(get_container)) -> dict:
    return container.metrics.snapshot()


@router.post("/admin/models/warmup", response_model=HealthRead)
def warmup(container: AppContainer = Depends(get_container)) -> HealthRead:
    try:
        container.engine.warmup()
    except InferenceUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return health(container)

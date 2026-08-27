from fastapi import APIRouter

from .attendance import router as attendance_router
from .employees import router as employees_router
from .enrollment import router as enrollment_router
from .health import router as health_router
from .recognition import router as recognition_router
from .reviews import router as reviews_router


api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(employees_router)
api_router.include_router(enrollment_router)
api_router.include_router(attendance_router)
api_router.include_router(reviews_router)
api_router.include_router(recognition_router)

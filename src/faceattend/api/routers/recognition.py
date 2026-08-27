from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..dependencies import get_container
from ...application.container import AppContainer
from ...application.recognition import process_frame
from ...domain.schemas import RecognitionRead


router = APIRouter(prefix="/recognition", tags=["recognition"])


@router.post("/frame", response_model=RecognitionRead)
def recognize_frame(
    image: UploadFile = File(...),
    client_id: str = Form(..., min_length=1, max_length=128),
    camera_id: str = Form("laptop-webcam", min_length=1, max_length=128),
    direction: str = Form("auto"),
    dry_run: bool = Form(False),
    container: AppContainer = Depends(get_container),
) -> RecognitionRead:
    return process_frame(container, image.file.read(), client_id, camera_id, direction, dry_run)

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routers import api_router
from .application.container import build_container


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = build_container()
    yield
    app.state.container.close()


app = FastAPI(title="FaceAttend", version="0.2.0", lifespan=lifespan)
web_root = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=web_root), name="static")
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    return FileResponse(web_root / "index.html")

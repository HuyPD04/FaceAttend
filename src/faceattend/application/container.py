from __future__ import annotations

from dataclasses import dataclass

from ..core.config import Settings, get_settings
from ..infrastructure.faiss_index import FaissTemplateIndex
from ..infrastructure.mongo_repository import MongoRepository
from ..services.attendance import AttendanceService, TemporalConfirmation
from ..services.crypto import EmbeddingCipher
from ..services.face_engine import FaceEngine
from ..services.liveness import LivenessService
from ..services.metrics import MetricsRegistry
from ..services.quality import FaceQualityGate


@dataclass
class AppContainer:
    settings: Settings
    repository: MongoRepository
    cipher: EmbeddingCipher
    index: FaissTemplateIndex
    engine: FaceEngine
    quality: FaceQualityGate
    liveness: LivenessService
    confirmation: TemporalConfirmation
    attendance: AttendanceService
    metrics: MetricsRegistry

    def close(self) -> None:
        self.repository.close()


def build_container() -> AppContainer:
    settings = get_settings()
    repository = MongoRepository(settings)
    repository.ensure_indexes()
    cipher = EmbeddingCipher(settings.fernet_key, settings.embedding_dimension)
    index = FaissTemplateIndex(settings.faiss_index_path, settings.embedding_dimension)
    index.rebuild(repository.active_template_rows(), cipher)
    return AppContainer(
        settings=settings,
        repository=repository,
        cipher=cipher,
        index=index,
        engine=FaceEngine(settings),
        quality=FaceQualityGate(settings),
        liveness=LivenessService(settings),
        confirmation=TemporalConfirmation(settings.confirmation_window_seconds),
        attendance=AttendanceService(repository, settings),
        metrics=MetricsRegistry(),
    )

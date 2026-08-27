from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from ..services.crypto import EmbeddingCipher


@dataclass(frozen=True)
class SearchCandidate:
    faiss_id: int
    employee_id: str
    employee_code: str
    full_name: str
    score: float


class FaissTemplateIndex:
    def __init__(self, index_path: Path, dimension: int):
        self._path = index_path
        self._dimension = dimension
        self._lock = threading.RLock()
        self._metadata: dict[int, dict[str, str]] = {}
        self._index = self._new_index()

    def _new_index(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))

    @property
    def size(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    def rebuild(self, rows: Iterable[dict], cipher: EmbeddingCipher) -> None:
        rows = list(rows)
        new_index = self._new_index()
        metadata: dict[int, dict[str, str]] = {}
        if rows:
            vectors = np.ascontiguousarray(
                np.stack([cipher.decrypt(bytes(row["embedding_encrypted"])) for row in rows]),
                dtype=np.float32,
            )
            identifiers = np.asarray([row["faiss_id"] for row in rows], dtype=np.int64)
            new_index.add_with_ids(vectors, identifiers)
            metadata = {
                int(row["faiss_id"]): {
                    "employee_id": str(row["employee_id"]),
                    "employee_code": row["employee_code"],
                    "full_name": row["full_name"],
                }
                for row in rows
            }
        with self._lock:
            self._index = new_index
            self._metadata = metadata
            self._persist_locked()

    def add(
        self,
        *,
        faiss_id: int,
        embedding: np.ndarray,
        employee_id: str,
        employee_code: str,
        full_name: str,
    ) -> None:
        vector = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        identifier = np.asarray([faiss_id], dtype=np.int64)
        with self._lock:
            self._index.add_with_ids(vector, identifier)
            self._metadata[faiss_id] = {
                "employee_id": employee_id,
                "employee_code": employee_code,
                "full_name": full_name,
            }
            self._persist_locked()

    def remove(self, faiss_ids: list[int]) -> None:
        if not faiss_ids:
            return
        identifiers = np.asarray(faiss_ids, dtype=np.int64)
        with self._lock:
            self._index.remove_ids(identifiers)
            for faiss_id in faiss_ids:
                self._metadata.pop(faiss_id, None)
            self._persist_locked()

    def search(self, embedding: np.ndarray, top_k: int = 8) -> list[SearchCandidate]:
        vector = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, identifiers = self._index.search(vector, min(top_k, self._index.ntotal))
            candidates: list[SearchCandidate] = []
            for score, faiss_id in zip(scores[0], identifiers[0], strict=True):
                metadata = self._metadata.get(int(faiss_id))
                if metadata is None or int(faiss_id) < 0:
                    continue
                candidates.append(
                    SearchCandidate(
                        faiss_id=int(faiss_id),
                        employee_id=metadata["employee_id"],
                        employee_code=metadata["employee_code"],
                        full_name=metadata["full_name"],
                        score=float(score),
                    )
                )
            return candidates

    def _persist_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(".tmp")
        faiss.write_index(self._index, str(temp_path))
        os.replace(temp_path, self._path)

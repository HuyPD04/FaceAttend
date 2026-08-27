from __future__ import annotations

import numpy as np
from cryptography.fernet import Fernet


class EmbeddingCipher:
    def __init__(self, key: bytes, dimension: int):
        self._fernet = Fernet(key)
        self._dimension = dimension

    def encrypt(self, embedding: np.ndarray) -> bytes:
        vector = self._normalise(embedding)
        return self._fernet.encrypt(vector.tobytes())

    def decrypt(self, payload: bytes) -> np.ndarray:
        vector = np.frombuffer(self.decrypt_bytes(payload), dtype=np.float32)
        if vector.shape != (self._dimension,):
            raise ValueError("Stored embedding has an unexpected dimension")
        return self._normalise(vector)

    def encrypt_bytes(self, payload: bytes) -> bytes:
        return self._fernet.encrypt(payload)

    def decrypt_bytes(self, payload: bytes) -> bytes:
        return self._fernet.decrypt(payload)

    def _normalise(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape != (self._dimension,):
            raise ValueError(
                f"Expected embedding dimension {self._dimension}, got {vector.shape[0]}"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("Embedding must not be a zero vector")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

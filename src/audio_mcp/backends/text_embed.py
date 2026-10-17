import threading

import numpy as np

from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo


class MiniLMEmbedder:
    def __init__(self, settings):
        self.settings = settings
        self.info = ModelInfo(stage="text_embedding", name="sentence-transformers", version="all-MiniLM-L6-v2")
        self._model = None
        self._lock = threading.Lock()

    def available(self):
        from audio_mcp.backends.base import _importable

        if not _importable("sentence_transformers"):
            return False, "sentence_transformers not installed (uv sync --extra models)"
        return True, None

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                d = self.settings.device
                dev = "cuda" if d == "cuda" or (d == "auto" and torch.cuda.is_available()) else "cpu"
                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev,
                                                  cache_folder=str(self.settings.resolved_models_dir / "st"))
            except Exception as e:
                raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, f"text embedder failed to load: {e}") from e

    def embed(self, texts: list[str]) -> np.ndarray:
        self.load()
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)

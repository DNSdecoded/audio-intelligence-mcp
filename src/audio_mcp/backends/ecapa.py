import threading

import numpy as np

from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo


class EcapaEmbedder:
    def __init__(self, settings):
        self.settings = settings
        self.info = ModelInfo(stage="diarization", name="speechbrain-ecapa", version="spkrec-ecapa-voxceleb")
        self._model = None
        self._lock = threading.Lock()

    def available(self):
        from audio_mcp.backends.base import _importable

        if not (_importable("speechbrain") and _importable("torch")):
            return False, "speechbrain not installed (uv sync --extra models)"
        return True, None

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy

                d = self.settings.device
                dev = "cuda" if d == "cuda" or (d == "auto" and torch.cuda.is_available()) else "cpu"
                # COPY avoids the symlink privilege error on Windows
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(self.settings.resolved_models_dir / "ecapa"),
                    run_opts={"device": dev}, local_strategy=LocalStrategy.COPY)
            except Exception as e:
                raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, f"ECAPA failed to load: {e}") from e

    def embed(self, windows) -> np.ndarray:
        self.load()
        import torch

        if not windows:
            return np.zeros((0, 192), dtype=np.float32)
        n = max(len(w) for w in windows)
        batch = np.zeros((len(windows), n), dtype=np.float32)
        for i, w in enumerate(windows):
            batch[i, : len(w)] = w
        lens = torch.tensor([len(w) / n for w in windows], dtype=torch.float32)
        out = []
        with torch.inference_mode():
            for i in range(0, len(windows), 32):
                e = self._model.encode_batch(torch.from_numpy(batch[i:i + 32]), lens[i:i + 32])
                out.append(e.squeeze(1).cpu().numpy())
        emb = np.concatenate(out)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

import csv
import shutil
import threading
from importlib import resources
from pathlib import Path

import httpx
import numpy as np

from audio_mcp.backends.base import TagResult
from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.ingest.decode import resample

CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
CKPT_NAME = "Cnn14_mAP=0.431.pth"
MIN_BYTES = 100_000_000
SR, WIN, HOP, BATCH = 32000, 1.0, 0.5, 16


def _labels() -> list[str]:
    with resources.files("audio_mcp.backends").joinpath("audioset_labels.csv").open(encoding="utf-8") as f:
        return [r["display_name"] for r in csv.DictReader(f)]


class PannsTagger:
    def __init__(self, settings):
        self.settings = settings
        self.info = ModelInfo(stage="events", name="panns-cnn14", version="mAP0.431")
        self._model = None
        self._lock = threading.Lock()

    def available(self):
        from audio_mcp.backends.base import _importable

        if not (_importable("panns_inference") and _importable("torch")):
            return False, "panns_inference not installed (uv sync --extra models)"
        return True, None

    def _checkpoint(self) -> Path:
        path = self.settings.resolved_models_dir / "panns" / CKPT_NAME
        if path.exists() and path.stat().st_size > MIN_BYTES:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        with httpx.stream("GET", CKPT_URL, follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as r:
            r.raise_for_status()
            with part.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        if part.stat().st_size <= MIN_BYTES:
            part.unlink()
            raise RuntimeError("downloaded checkpoint is too small")
        part.replace(path)
        return path

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            try:
                # panns_inference's config.py shells out to wget (absent on Windows) unless this file exists
                labels = Path.home() / "panns_data" / "class_labels_indices.csv"
                if not labels.exists():
                    labels.parent.mkdir(parents=True, exist_ok=True)
                    with resources.as_file(resources.files("audio_mcp.backends").joinpath("audioset_labels.csv")) as src:
                        shutil.copyfile(src, labels)
                ckpt = self._checkpoint()
                import torch
                from panns_inference import AudioTagging

                d = self.settings.device
                cuda = d == "cuda" or (d == "auto" and torch.cuda.is_available())
                self._model = AudioTagging(checkpoint_path=str(ckpt), device="cuda" if cuda else "cpu")
            except Exception as e:
                raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, f"panns failed to load: {e}") from e

    def tag_windows(self, mono16k) -> TagResult:
        self.load()
        import torch

        x = resample(np.asarray(mono16k, dtype=np.float32), 16000, SR)
        win, hop = int(WIN * SR), int(HOP * SR)
        n = 1 if len(x) <= win else (len(x) - win + hop - 1) // hop + 1
        scores, embs = [], []
        with torch.inference_mode():
            for b in range(0, n, BATCH):
                frames = np.zeros((min(BATCH, n - b), win), dtype=np.float32)  # only one batch in memory
                for k in range(len(frames)):
                    seg = x[(b + k) * hop : (b + k) * hop + win]
                    frames[k, : len(seg)] = seg  # zero-pads a short last window
                s, e = self._model.inference(frames)
                scores.append(s)
                embs.append(e)
        return TagResult(
            labels=_labels(),
            scores=np.concatenate(scores),
            window_starts=np.arange(n) * HOP,
            window_seconds=WIN,
            embeddings=np.concatenate(embs),
        )

    def embed(self, mono16k) -> np.ndarray:
        return self.tag_windows(mono16k).embeddings.mean(axis=0)

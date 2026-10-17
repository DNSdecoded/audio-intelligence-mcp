import os
import threading

import numpy as np

from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.types import Interval, SpeakerTurn


class PyannoteDiarizer:
    """Optional backend: needs the pyannote extra and an HF_TOKEN with the model's terms accepted."""

    def __init__(self, settings):
        self.settings = settings
        self.info = ModelInfo(stage="diarization", name="pyannote", version="speaker-diarization-3.1")
        self._pipeline = None
        self._lock = threading.Lock()

    def available(self):
        from audio_mcp.backends.base import _importable

        if not _importable("pyannote.audio"):
            return False, "pyannote.audio not installed (uv sync --extra pyannote)"
        if not os.environ.get("HF_TOKEN"):
            return False, "HF_TOKEN not set (required for the gated pyannote model)"
        return True, None

    def load(self):
        with self._lock:
            if self._pipeline is not None:
                return
            try:
                from pyannote.audio import Pipeline

                self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                          use_auth_token=os.environ["HF_TOKEN"])
            except Exception as e:
                raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, f"pyannote failed to load: {e}") from e

    def diarize(self, mono16k, num_speakers):
        self.load()
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(mono16k, dtype=np.float32))[None]
        ann = self._pipeline({"waveform": wav, "sample_rate": 16000}, num_speakers=num_speakers)
        names: dict = {}
        turns = []
        for seg, _, label in ann.itertracks(yield_label=True):
            names.setdefault(label, f"speaker_{len(names) + 1}")
            turns.append(SpeakerTurn(speaker_id=names[label], start=seg.start, end=seg.end))
        overlaps = [Interval(start=s.start, end=s.end) for s in ann.get_overlap()]
        return turns, overlaps

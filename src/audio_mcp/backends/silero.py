import numpy as np

from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.types import Interval


class SileroVAD:
    info = ModelInfo(stage="vad", name="silero-vad", version="5")

    def __init__(self):
        self._model = None

    def available(self):
        try:
            import silero_vad  # noqa: F401
        except ImportError:
            return False, "silero_vad not installed (uv sync --extra models)"
        return True, None

    def load(self):
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=True)
        except Exception as e:
            raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, f"silero-vad failed to load: {e}") from e

    def speech_regions(self, mono16k: np.ndarray) -> list[Interval]:
        import torch
        from silero_vad import get_speech_timestamps

        self.load()
        x = torch.from_numpy(np.ascontiguousarray(mono16k, dtype=np.float32))
        ts = get_speech_timestamps(x, self._model, sampling_rate=16000, return_seconds=True)
        return [Interval(start=t["start"], end=t["end"]) for t in ts]

import numpy as np

from audio_mcp.dsp.features import frame_rms
from audio_mcp.envelope import ModelInfo
from audio_mcp.types import Interval

_FRAME = 0.03
_MIN_DB, _FLOOR_MARGIN_DB = -40.0, 12.0
_MERGE_GAP, _MIN_REGION = 0.3, 0.25


class EnergyVAD:
    info = ModelInfo(stage="vad", name="energy-vad", version="1")

    def available(self):
        return True, None

    def load(self):
        pass

    def speech_regions(self, mono16k: np.ndarray) -> list[Interval]:
        rms = frame_rms(mono16k, 16000, win=_FRAME, hop=_FRAME)
        db = 20 * np.log10(np.maximum(rms, 1e-10))
        speech = db > max(_MIN_DB, np.percentile(db, 10) + _FLOOR_MARGIN_DB)
        regions: list[list[float]] = []
        for i in np.flatnonzero(speech):
            s, e = i * _FRAME, (i + 1) * _FRAME
            if regions and s - regions[-1][1] < _MERGE_GAP:
                regions[-1][1] = e
            else:
                regions.append([s, e])
        return [Interval(start=s, end=e) for s, e in regions if e - s >= _MIN_REGION]

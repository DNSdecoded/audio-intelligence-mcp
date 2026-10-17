import pytest
import soundfile as sf

from audio_mcp.analysis.events import EVENT_CATEGORIES, frames_to_events
from audio_mcp.backends.panns import PannsTagger
from audio_mcp.ingest.decode import resample

pytestmark = pytest.mark.slow


def _events(settings, path, threshold):
    settings.backend_profile = "real"
    x, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        x, sr = resample(x, sr, 16000), 16000
    return frames_to_events(PannsTagger(settings).tag_windows(x), duration=len(x) / sr, threshold=threshold)


def test_alarm_detected(settings, synth):
    ev = [e for e in _events(settings, synth["alarm"], 0.2) if e.label in EVENT_CATEGORIES["alarm"]]
    assert ev and min(abs(e.start - 1.0) for e in ev) <= 0.6


def test_silence_has_no_events(settings, synth):
    assert {e.label for e in _events(settings, synth["silence"], 0.2)} <= {"Silence"}

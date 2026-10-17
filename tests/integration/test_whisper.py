import pytest
import soundfile as sf

from audio_mcp.backends.whisper import FasterWhisperASR

pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("ignore:.*symlinks:UserWarning")]


def test_transcribes_reference(settings, speech_en):
    settings.backend_profile = "real"
    x, _ = sf.read(speech_en, dtype="float32")
    r = FasterWhisperASR(settings).transcribe(x)
    text = " ".join(s.text for s in r.segments).lower()
    assert "report" in text and "friday" in text and r.language == "en"
    assert all(0 <= w.confidence <= 1 for s in r.segments for w in s.words)
    assert r.segments[0].start < 1.0 and r.segments[-1].end <= len(x) / 16000 + 0.5
    assert r.languages[0]["end"] > 0

import pytest
import soundfile as sf

from audio_mcp.backends.silero import SileroVAD

pytestmark = pytest.mark.slow


def test_silero_covers_speech(speech_en):
    x, sr = sf.read(speech_en, dtype="float32")
    assert sr == 16000
    r = SileroVAD().speech_regions(x)
    assert sum(i.end - i.start for i in r) >= 0.6 * len(x) / sr

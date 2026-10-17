import pytest
import soundfile as sf

from conftest import FIXTURES_DIR
from audio_mcp.analysis.diarize import cluster_speakers
from audio_mcp.backends.ecapa import EcapaEmbedder
from audio_mcp.backends.silero import SileroVAD
from audio_mcp.eval.metrics import der, load_rttm

pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("ignore::UserWarning"),
              pytest.mark.filterwarnings("ignore::FutureWarning")]


def test_ecapa_two_speakers(settings):
    x, _ = sf.read(FIXTURES_DIR / "dialogue_2spk.wav", dtype="float32")
    turns = cluster_speakers(x, SileroVAD().speech_regions(x), EcapaEmbedder(settings),
                             threshold=settings.diarization_threshold, num_speakers=None, max_speakers=8)
    assert len({t.speaker_id for t in turns}) == 2
    hyp = [t.model_dump() for t in turns]
    assert der(load_rttm(FIXTURES_DIR / "dialogue_2spk.rttm"), hyp)["der"] < 0.35

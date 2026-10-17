import os
from pathlib import Path

import platformdirs
import pytest

from audio_mcp.config import Settings
from fixtures.synth import make_synthetic

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth"))


@pytest.fixture
def speech_en():
    return FIXTURES_DIR / "speech_en.wav"


@pytest.fixture
def dialogue():
    return FIXTURES_DIR / "dialogue_2spk.wav"


@pytest.fixture
def speech_es():
    return FIXTURES_DIR / "speech_es.ogg"


@pytest.fixture
def settings(tmp_path, synth):
    # persistent shared model cache so slow tests don't re-download; everything else stays in tmp
    models = Path(os.environ.get("AUDIO_MCP_TEST_MODELS_DIR") or platformdirs.user_data_path("audio-mcp") / "models")
    models.mkdir(parents=True, exist_ok=True)
    s = Settings(
        models_dir=models,
        allowed_roots=[tmp_path, FIXTURES_DIR, synth["sine"].parent],
        data_dir=tmp_path / "data",
        backend_profile="fake",
    )
    s.resolved_temp_dir.mkdir(parents=True, exist_ok=True)
    return s


def load(path):
    import soundfile as sf

    x, sr = sf.read(path, always_2d=True)
    return x.T.astype("float32"), sr

from pathlib import Path

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
    s = Settings(
        allowed_roots=[tmp_path, FIXTURES_DIR, next(iter(synth.values())).parent],
        data_dir=tmp_path / "data",
        backend_profile="fake",
    )
    s.resolved_temp_dir.mkdir(parents=True, exist_ok=True)
    return s

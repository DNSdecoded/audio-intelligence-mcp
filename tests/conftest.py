from pathlib import Path

import pytest

from audio_mcp.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        allowed_roots=[tmp_path, FIXTURES_DIR],
        data_dir=tmp_path / "data",
        backend_profile="fake",
    )

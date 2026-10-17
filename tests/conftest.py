from pathlib import Path

import pytest

from audio_mcp.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture
def settings(tmp_path):
    s = Settings(
        allowed_roots=[tmp_path, FIXTURES_DIR],
        data_dir=tmp_path / "data",
        backend_profile="fake",
    )
    s.resolved_temp_dir.mkdir(parents=True, exist_ok=True)
    return s

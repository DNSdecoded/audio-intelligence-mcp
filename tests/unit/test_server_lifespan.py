import sys

import pytest

from audio_mcp import server


@pytest.mark.asyncio
async def test_lifespan_routes_stray_prints_to_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIO_MCP_BACKEND_PROFILE", "fake")
    monkeypatch.setenv("AUDIO_MCP_DATA_DIR", str(tmp_path))
    before = sys.stdout
    async with server.lifespan(None):
        print("library noise")
        assert sys.stdout is sys.stderr
    assert sys.stdout is before
    out = capsys.readouterr()
    assert "library noise" not in out.out and "library noise" in out.err

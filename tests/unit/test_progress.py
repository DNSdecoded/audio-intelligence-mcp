import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from audio_mcp import server


@pytest.mark.asyncio
async def test_sync_call_sends_monotonic_progress(monkeypatch, tmp_path, speech_en):
    monkeypatch.setenv("AUDIO_MCP_BACKEND_PROFILE", "fake")
    monkeypatch.setenv("AUDIO_MCP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AUDIO_MCP_ALLOWED_ROOTS", json.dumps([str(speech_en.parent)]))
    seen = []

    async def on_progress(progress, total, message):
        seen.append(progress)

    async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
        res = await client.call_tool(
            "transcribe_audio", {"source": {"type": "file_path", "path": str(speech_en)}},
            progress_callback=on_progress)
    assert not res.isError
    assert len(seen) >= 2 and seen == sorted(set(seen))

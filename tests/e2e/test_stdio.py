import json
import os
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from conftest import FIXTURES_DIR

EXPECTED_TOOLS = {"analyze_audio", "transcribe_audio", "detect_audio_events", "extract_audio_features",
                  "inspect_audio_quality", "get_audio_timeline", "ask_audio", "summarize_audio", "compare_audio", "search_audio_library"}


@pytest.mark.asyncio
async def test_stdio_roundtrip(tmp_path):
    params = StdioServerParameters(command=sys.executable, args=["-m", "audio_mcp"], env={
        **os.environ, "AUDIO_MCP_BACKEND_PROFILE": "fake", "AUDIO_MCP_DATA_DIR": str(tmp_path),
        "AUDIO_MCP_ALLOWED_ROOTS": json.dumps([str(FIXTURES_DIR)])})
    with anyio.fail_after(60):
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            assert {t.name for t in tools} == EXPECTED_TOOLS
            assert all(t.inputSchema["type"] == "object" for t in tools)
            res = await s.call_tool("transcribe_audio", {"source": {"type": "file_path", "path": str(FIXTURES_DIR / "speech_en.wav")}})
            assert json.loads(res.content[0].text)["status"] == "success"

import json
import os
import socket
import subprocess
import sys

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from conftest import FIXTURES_DIR


@pytest.mark.asyncio
async def test_http_server_serves_tools_with_token(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "AUDIO_MCP_TRANSPORT": "streamable-http", "AUDIO_MCP_HTTP_PORT": str(port),
           "AUDIO_MCP_HTTP_TOKEN": "tok", "AUDIO_MCP_BACKEND_PROFILE": "fake",
           "AUDIO_MCP_DATA_DIR": str(tmp_path / "data"), "AUDIO_MCP_ALLOWED_ROOTS": json.dumps([str(FIXTURES_DIR)])}
    env.pop("AUDIO_MCP_CONFIG", None)
    proc = subprocess.Popen([sys.executable, "-m", "audio_mcp"], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        with anyio.fail_after(60):
            async with httpx.AsyncClient(timeout=2) as probe:
                while True:  # wait for the listener without blocking the event loop
                    try:
                        assert (await probe.post(url)).status_code == 401  # no token
                        break
                    except httpx.TransportError:
                        assert proc.poll() is None, "server exited"
                        await anyio.sleep(0.3)
            async with streamablehttp_client(url, headers={"Authorization": "Bearer tok"}) as (r, w, _):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    assert "transcribe_audio" in {t.name for t in (await sess.list_tools()).tools}
                    res = await sess.call_tool("transcribe_audio", {"source": {
                        "type": "file_path", "path": str(FIXTURES_DIR / "speech_en.wav")}})
                    assert not res.isError and json.loads(res.content[0].text)["status"] == "success"
                    res = await sess.call_tool("transcribe_audio", {"source": {
                        "type": "file_path", "path": str(FIXTURES_DIR / "speech_en.wav")}, "hotwords": ["x" * 101]})
                    assert json.loads(res.content[0].text)["status"] == "error"  # over-long hotword rejected by the schema
    finally:
        proc.terminate()
        proc.wait(timeout=20)

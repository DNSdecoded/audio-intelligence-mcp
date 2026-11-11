import json

import pytest
from starlette.testclient import TestClient

from audio_mcp import server

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
HDRS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIO_MCP_BACKEND_PROFILE", "fake")
    monkeypatch.setenv("AUDIO_MCP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AUDIO_MCP_ALLOWED_ROOTS", json.dumps([str(tmp_path)]))
    monkeypatch.setenv("AUDIO_MCP_TRANSPORT", "streamable-http")


def test_bearer_auth(env):
    # a fresh FastMCP session manager per app: streamable_http_app() can only run once per FastMCP instance
    server.mcp._session_manager = None
    app = server.http_app("s3cret")
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        assert c.post("/mcp", json=INIT, headers=HDRS).status_code == 401
        assert c.post("/mcp", json=INIT, headers={**HDRS, "Authorization": "Bearer wrong"}).status_code == 401
        ok = c.post("/mcp", json=INIT, headers={**HDRS, "Authorization": "Bearer s3cret"})
        assert ok.status_code == 200 and "audio-intelligence" in ok.text


@pytest.fixture(autouse=True)
def no_real_server(monkeypatch):
    monkeypatch.delenv("AUDIO_MCP_CONFIG", raising=False)  # a developer config must not change the guards
    import uvicorn

    def boom(*a, **k):
        raise AssertionError("uvicorn.run must not be reached")

    monkeypatch.setattr(uvicorn, "run", boom)


def test_main_starts_uvicorn_with_settings(env, monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    monkeypatch.setenv("AUDIO_MCP_HTTP_TOKEN", "x")
    monkeypatch.setenv("AUDIO_MCP_HTTP_PORT", "9123")
    server.mcp._session_manager = None
    server.main()
    assert calls == [{"host": "127.0.0.1", "port": 9123, "log_level": "warning"}]  # allowed_roots from env counts as explicit


def test_refuses_without_token(env, monkeypatch):
    monkeypatch.delenv("AUDIO_MCP_HTTP_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="bearer token"):
        server.main()


def test_refuses_default_allowed_roots(env, monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_HTTP_TOKEN", "x")
    monkeypatch.delenv("AUDIO_MCP_ALLOWED_ROOTS")
    with pytest.raises(SystemExit, match="allowed_roots"):
        server.main()


def test_lifespan_shared_across_sessions(env):
    import asyncio

    async def go():
        async with server.lifespan(None):
            first = server._app
            async with server.lifespan(None):
                assert server._app is first
            assert server._app is first  # inner session ending does not tear the app down
        assert server._app is None

    asyncio.run(go())

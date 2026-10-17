import base64
import json
import os
import shutil
import sys
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from conftest import FIXTURES_DIR

EXPECTED_TOOLS = {"analyze_audio", "transcribe_audio", "detect_audio_events", "extract_audio_features",
                  "inspect_audio_quality", "get_audio_timeline", "ask_audio", "summarize_audio", "compare_audio",
                  "search_audio_library", "get_job", "analyze_audio_stream", "export_audio_analysis"}


@asynccontextmanager
async def session(tmp_path, roots, profile="fake"):
    params = StdioServerParameters(command=sys.executable, args=["-m", "audio_mcp"], env={
        **os.environ, "AUDIO_MCP_BACKEND_PROFILE": profile, "AUDIO_MCP_DATA_DIR": str(tmp_path / "data"),
        "AUDIO_MCP_ALLOWED_ROOTS": json.dumps([str(r) for r in roots])})
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        yield s


async def call(s, name, args):
    res = await s.call_tool(name, args)
    assert not res.isError, (name, res.content[0].text)
    return json.loads(res.content[0].text)


@pytest.fixture
def samples_dir(tmp_path, synth):
    d = tmp_path / "samples"
    d.mkdir()
    for p in (FIXTURES_DIR / "speech_en.wav", FIXTURES_DIR / "dialogue_2spk.wav", synth["alarm"], synth["sine"]):
        shutil.copy(p, d / p.name)
    return d


@pytest.mark.asyncio
async def test_stdio_roundtrip(tmp_path):
    with anyio.fail_after(60):
        async with session(tmp_path, [FIXTURES_DIR]) as s:
            tools = (await s.list_tools()).tools
            assert {t.name for t in tools} == EXPECTED_TOOLS
            assert all(t.inputSchema["type"] == "object" and t.description for t in tools)
            out = await call(s, "transcribe_audio", {"source": {"type": "file_path",
                                                                 "path": str(FIXTURES_DIR / "speech_en.wav")}})
            assert out["status"] == "success"


@pytest.mark.asyncio
async def test_all_tools_invocable(tmp_path, samples_dir):
    def src(name):
        return {"type": "file_path", "path": str(samples_dir / name)}

    with anyio.fail_after(240):
        async with session(tmp_path, [samples_dir]) as s:
            outs = {}
            outs["analyze_audio"] = a = await call(s, "analyze_audio", {"source": src("dialogue_2spk.wav"),
                                                                        "goal": "meeting"})
            aid = a["audio_id"]
            by_id = {"type": "audio_id", "audio_id": aid}
            outs["transcribe_audio"] = await call(s, "transcribe_audio", {"source": by_id, "diarize": True})
            outs["detect_audio_events"] = await call(s, "detect_audio_events", {"source": src("alarm.wav"),
                                                                                "categories": ["alarm"]})
            outs["extract_audio_features"] = await call(s, "extract_audio_features", {"source": by_id})
            outs["inspect_audio_quality"] = await call(s, "inspect_audio_quality", {"source": src("sine.wav")})
            outs["get_audio_timeline"] = await call(s, "get_audio_timeline", {"source": by_id})
            outs["ask_audio"] = await call(s, "ask_audio", {"source": by_id,
                                                            "question": "When does someone mention the report?"})
            outs["summarize_audio"] = await call(s, "summarize_audio", {"source": by_id})
            outs["compare_audio"] = await call(s, "compare_audio", {"sources": [src("sine.wav"), src("alarm.wav")]})
            outs["search_audio_library"] = await call(s, "search_audio_library", {"action": "index",
                                                                                  "paths": [str(samples_dir)], "async_mode": "never"})
            job = await call(s, "transcribe_audio", {"source": src("speech_en.wav"), "async_mode": "always"})
            assert job["status"] == "processing"
            outs["get_job"] = await call(s, "get_job", {"job_id": job["job_id"]})
            start = await call(s, "analyze_audio_stream", {"action": "start", "chunk_format": "file"})
            sid = start["result"]["session_id"]
            await call(s, "analyze_audio_stream", {"action": "push", "session_id": sid,
                                                   "data": base64.b64encode((samples_dir / "speech_en.wav")
                                                                            .read_bytes()).decode()})
            outs["analyze_audio_stream"] = await call(s, "analyze_audio_stream", {
                "action": "finish", "session_id": sid, "tasks": ["transcription"]})
            outs["export_audio_analysis"] = await call(s, "export_audio_analysis", {"audio_id": aid, "format": "srt"})

    assert set(outs) == EXPECTED_TOOLS
    for name, out in outs.items():
        assert out["status"] in {"success", "partial", "processing"}, (name, out)
        assert out["processing"]["schema_version"] == "1.0", name
        assert out["processing"]["data_sent_off_machine"] is False, name
    ans = outs["ask_audio"]["result"]
    assert ans["support"] == "direct" and ans["evidence"][0]["start"] < ans["evidence"][0]["end"]
    assert outs["analyze_audio"]["processing"]["plan"]
    assert outs["search_audio_library"]["result"]["indexed"]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_models_keep_stdout_clean(tmp_path, samples_dir):
    """Model libraries that print on load (panns_inference does) must not corrupt the JSON-RPC stream."""
    with anyio.fail_after(600):
        async with session(tmp_path, [samples_dir], profile="real") as s:
            out = await call(s, "detect_audio_events", {"source": {"type": "file_path",
                                                                    "path": str(samples_dir / "alarm.wav")}})
            assert out["status"] == "success" and out["result"]["events"]

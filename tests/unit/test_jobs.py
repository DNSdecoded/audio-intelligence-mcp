import threading

import pytest

from conftest import fp, slow_transcribe, wait_until
from fixtures.synth import make_long
from audio_mcp.jobs import JobManager
from audio_mcp.tools.handlers import analyze_audio, extract_audio_features, get_job, transcribe_audio
from audio_mcp.tools.schemas import AnalyzeRequest, FeaturesRequest, GetJobRequest, TranscribeRequest


def status(app, jid):
    return get_job(app, GetJobRequest(job_id=jid))


def test_async_roundtrip(app, speech_en):
    r = transcribe_audio(app, TranscribeRequest(source=fp(speech_en), async_mode="always"))
    assert r["status"] == "processing" and r["audio_id"].startswith("aud_")
    jid = r["job_id"]
    wait_until(lambda: status(app, jid)["result"]["job_status"] == "done")
    s = status(app, jid)
    assert s["status"] == "success" and s["result"]["progress"] == 1.0
    assert s["result"]["response"]["status"] == "success" and s["result"]["response"]["result"]["segments"]


def test_cancel(app, speech_en, monkeypatch):
    monkeypatch.setattr(app.backends.asr, "transcribe", slow_transcribe(1.5))
    jid = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["transcription", "acoustic_features"],
                                            async_mode="always"))["job_id"]
    wait_until(lambda: status(app, jid)["result"]["job_status"] == "running")
    get_job(app, GetJobRequest(job_id=jid, action="cancel"))
    wait_until(lambda: status(app, jid)["result"]["job_status"] == "cancelled")


def test_queued_cancel_is_immediate():
    jm = JobManager(1)
    gate = threading.Event()
    first = jm.submit("t", lambda ev, prog: (gate.wait(5), {"status": "success"})[1])
    second = jm.submit("t", lambda ev, prog: {"status": "success"})
    assert jm.cancel(second).status == "cancelled"
    gate.set()
    wait_until(lambda: jm.get(first).status == "done")
    jm.shutdown()


def test_stage_timeout(app, speech_en, monkeypatch):
    app.settings.stage_timeout_seconds = 0.2
    monkeypatch.setattr(app.backends.asr, "transcribe", slow_transcribe(1.0))
    r = transcribe_audio(app, TranscribeRequest(source=fp(speech_en)))
    assert r["errors"][0]["code"] == "TIMEOUT" and r["errors"][0]["retryable"] is True


def test_unknown_job(app):
    assert status(app, "job_000000000000")["error"]["code"] == "JOB_NOT_FOUND"


def test_auto_stays_sync_for_short_audio(app, speech_en):
    assert extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["status"] == "success"


@pytest.mark.long
def test_long_file_goes_async(app, tmp_path):
    app.settings.allowed_roots.append(tmp_path)
    r = extract_audio_features(app, FeaturesRequest(source=fp(make_long(tmp_path))))
    assert r["status"] == "processing"
    wait_until(lambda: status(app, r["job_id"])["result"]["job_status"] == "done", timeout=600)


def test_sync_calls_respect_max_concurrent_jobs(app, speech_en, dialogue, monkeypatch):  # I3
    import time

    assert app.settings.max_concurrent_jobs == 1
    monkeypatch.setattr(app.backends.asr, "transcribe", slow_transcribe(0.8))
    out = []

    def call(p):
        out.append(transcribe_audio(app, TranscribeRequest(source=fp(p), async_mode="never"))["status"])

    t0 = time.monotonic()
    ts = [threading.Thread(target=call, args=(p,)) for p in (speech_en, dialogue)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert out == ["success", "success"] and time.monotonic() - t0 >= 1.6


def test_nested_run_analysis_does_not_deadlock(app, speech_en):  # I3: plan_auto runs inside analyze_audio
    out = []
    t = threading.Thread(target=lambda: out.append(analyze_audio(app, AnalyzeRequest(
        source=fp(speech_en), tasks="auto", async_mode="never"))))
    t.start()
    t.join(30)
    assert out and out[0]["status"] == "success"


@pytest.mark.parametrize("tool,model,params", [
    ("inspect_audio_quality", "InspectQualityRequest", {}),
    ("get_audio_timeline", "TimelineRequest", {}),
    ("ask_audio", "AskRequest", {"question": "what is said?"}),
    ("summarize_audio", "SummarizeRequest", {}),
    ("compare_audio", "CompareRequest", {"sources": "two"}),
    ("export_audio_analysis", "ExportRequest", {"format": "json"}),
    ("search_audio_library", "SearchLibraryRequest", {"action": "index"}),
])
def test_new_async_capable_tools(app, speech_en, dialogue, tool, model, params):  # I4
    from audio_mcp.tools import handlers, schemas

    p = dict(params)
    if tool == "compare_audio":
        p["sources"] = [fp(speech_en), fp(dialogue)]
    elif tool == "export_audio_analysis":
        p["audio_id"] = app.store.ingest(schemas.FeaturesRequest(source=fp(speech_en)).source).audio_id
    elif tool == "search_audio_library":
        p["paths"] = [str(speech_en)]
    else:
        p["source"] = fp(speech_en)
    r = getattr(handlers, tool)(app, getattr(schemas, model)(**p, async_mode="always"))
    assert r["status"] == "processing", r
    wait_until(lambda: status(app, r["job_id"])["result"]["job_status"] != "running"
               and status(app, r["job_id"])["result"]["job_status"] != "queued", timeout=60)
    s = status(app, r["job_id"])["result"]
    assert s["job_status"] == "done" and s["response"]["status"] == "success", s


def test_auto_async_for_multi_source_tools(app, speech_en, tmp_path):  # I4
    from audio_mcp.tools.handlers import compare_audio, search_audio_library
    from audio_mcp.tools.schemas import CompareRequest, SearchLibraryRequest

    r = search_audio_library(app, SearchLibraryRequest(action="index", paths=[str(tmp_path)]))
    assert r["status"] == "processing"
    r = compare_audio(app, CompareRequest(sources=[fp(speech_en)] * 2, baseline=[fp(speech_en)] * 2))
    assert r["status"] == "processing"
    assert compare_audio(app, CompareRequest(sources=[fp(speech_en)] * 2))["status"] == "success"

"""Regression tests for confirmed audit findings (BUG-01 through BUG-11)."""
import asyncio
import base64
import threading
from pathlib import Path

import anyio
import numpy as np
import pytest

from conftest import fp, load
from audio_mcp.analysis.diarize import cluster_speakers
from audio_mcp.backends.fake import FakeSpeaker
from audio_mcp.backends.llm import is_loopback
from audio_mcp.config import Settings
from audio_mcp.dsp.features import extract_features
from audio_mcp.dsp.quality import diagnose_quality
from audio_mcp.envelope import AudioMCPError, ErrorCode
from audio_mcp.ingest.sources import FilePathSource, UrlSource, _blocked, materialize_source
from audio_mcp.jobs import JOB_CONTEXT
from audio_mcp.library.index import Library
from audio_mcp.pipeline import runner as runner_mod
from audio_mcp.pipeline.runner import run_analysis
from audio_mcp.pipeline.stages import STAGES, Stage
from audio_mcp.reasoning.ask import _tokens
from audio_mcp.reasoning.llm_grounded import _parse
from audio_mcp.reasoning.summarize import summarize
from audio_mcp.store.retention import _pid_alive
from audio_mcp.tools.handlers import (
    analyze_audio,
    analyze_audio_stream,
    export_audio_analysis,
    extract_audio_features,
)
from audio_mcp.tools.schemas import (
    AnalyzeRequest,
    ExportRequest,
    FeaturesRequest,
    StreamRequest,
)
from audio_mcp.types import Interval


def test_bug01_hum_not_flagged_on_pure_440hz_tones(synth):
    """BUG-01: Pure 440 Hz tones (low_level, sine, clipped) have ~0 power at 50/60 Hz and must not flag HUM."""
    for name in ("low_level", "sine", "clipped"):
        x, sr = load(synth[name])
        q = diagnose_quality(x, sr, extract_features(x, sr)["values"], [])
        codes = {i["code"] for i in q["issues"]}
        assert "HUM" not in codes, f"False HUM flagged on {name}: {q['metrics']['hum_excess_db']}"


def test_bug02_analyze_options_respected_and_none_accepted(app, dialogue):
    """BUG-02: AnalyzeOptions.identify_speakers, include_evidence, timestamps, and options=None must work."""
    req_none = AnalyzeRequest(source=fp(dialogue), options=None)
    assert req_none.options.timestamps is True

    # identify_speakers=False in auto mode must not run speaker_diarization
    r_no_spk = analyze_audio(
        app,
        AnalyzeRequest(
            source=fp(dialogue),
            tasks="auto",
            goal="meeting",
            options={"identify_speakers": False},
        ),
    )
    assert "speakers" not in r_no_spk["analysis"]

    # include_evidence=False must strip evidence_ids from summary/uncertainty
    r_no_ev = analyze_audio(
        app,
        AnalyzeRequest(
            source=fp(dialogue),
            tasks=["transcription", "summarization", "uncertainty"],
            options={"include_evidence": False},
        ),
    )
    assert "evidence_ids" not in repr(r_no_ev["analysis"])

    # timestamps=False must strip start/end timestamps and words from transcription segments
    r_no_ts = analyze_audio(
        app,
        AnalyzeRequest(
            source=fp(dialogue),
            tasks=["transcription"],
            options={"timestamps": False},
        ),
    )
    segs = r_no_ts["analysis"]["transcription"]["segments"]
    assert segs and all("start" not in s and "end" not in s and "words" not in s for s in segs)


def test_bug03_multilingual_tokens_and_fts_combining_marks(tmp_path):
    """BUG-03: _tokens and Library._fts must preserve non-Latin scripts and combining vowel marks."""
    assert "नमस्ते" in _tokens("नमस्ते दुनिया")
    assert "दुनिया" in _tokens("नमस्ते दुनिया")
    assert "mañana" in _tokens("reunión mañana")

    segs = [
        {"id": "seg_0001", "start": 0.0, "end": 2.0, "text": "नमस्ते दुनिया आज बजट बैठक है"},
        {"id": "seg_0002", "start": 2.0, "end": 4.0, "text": "बजट रिपोर्ट शुक्रवार तक पूरी करें"},
    ]
    summ = summarize({"transcription": {"segments": segs}}, 4.0, None, length="short", focus="बजट")
    assert summ["topics"], "Multilingual summary must extract non-empty topics"

    lib = Library(tmp_path / "lib.sqlite", None)
    try:
        audio_file = tmp_path / "hi.wav"
        audio_file.write_bytes(b"RIFF0000WAVE")
        rec = type("Rec", (), {"audio_id": "aud_0123456789abcdef", "probe": type("P", (), {"duration_seconds": 4.0})()})()
        lib.add(rec, audio_file, {"transcription": {"segments": segs, "language": "hi"}}, None)
        matches = lib.search("नमस्ते")
        assert matches and matches[0]["audio_id"] == "aud_0123456789abcdef"
    finally:
        lib.close()


def test_bug04_abandoned_thread_cannot_poison_next_stage_cache(app, speech_en, monkeypatch):
    """BUG-04: An abandoned timed-out stage thread must not mutate StageContext of later stages."""
    release, mutated = threading.Event(), threading.Event()

    def slow_leaky(ctx):
        release.wait(5)  # outlives its timeout, then mutates while the next stage is running
        ctx.warnings.append({"code": "GHOST_WARNING", "message": "from abandoned thread"})
        ctx.no_cache = True
        mutated.set()
        return {"events": [], "threshold": 0.3}

    def next_stage(ctx):
        release.set()
        assert mutated.wait(5), "abandoned thread never mutated its context"
        return {"units": {}, "values": {"rms": 0.1}}

    monkeypatch.setitem(STAGES, "events", Stage("events", (), slow_leaky, lambda a, o: {"k": 1}))
    monkeypatch.setitem(STAGES, "features", Stage("features", (), next_stage, lambda a, o: {"k": 2}))
    real = runner_mod._call_with_timeout  # short timeout for events only, long for features, no shared settings
    monkeypatch.setattr(runner_mod, "_call_with_timeout",
                        lambda fn, ctx, timeout: real(fn, ctx, 0.08 if fn is slow_leaky else 2.0))
    src = FilePathSource(type="file_path", path=str(speech_en))
    o = run_analysis(app, src, ["events", "features"], {})
    assert any(e["stage"] == "events" and e["code"] == "TIMEOUT" for e in o.errors)
    assert "features" in o.results
    assert all(w["code"] != "GHOST_WARNING" for w in o.warnings)
    o2 = run_analysis(app, src, ["features"], {})
    assert "features" in o2.processing.cache_hits
    assert all(w["code"] != "GHOST_WARNING" for w in o2.warnings)


def test_bug05_diarization_handles_zero_norm_windows():
    """BUG-05: cluster_speakers must not crash with ValueError when a speech window is all zeros."""
    x = np.zeros(16000 * 4, dtype=np.float32)
    x[:16000 * 2] = 0.2 * np.sin(2 * np.pi * 200 * np.arange(16000 * 2) / 16000)
    regions = [Interval(start=0.0, end=1.5), Interval(start=2.0, end=3.5)]
    turns = cluster_speakers(x, regions, FakeSpeaker(), threshold=0.55, num_speakers=None, max_speakers=8)
    assert len(turns) == 2


def test_bug06_stream_word_offsets_empty_chunk_and_redacted_reconcile(app, speech_en):
    """BUG-06: Stream _asr_window word offsets, empty chunk handling, partial file init, and redacted reconcile."""
    sid = analyze_audio_stream(
        app,
        StreamRequest(action="start", chunk_format="pcm", sample_rate_hz=16000, channels=1, sample_format="s16le"),
    )["result"]["session_id"]
    empty_res = analyze_audio_stream(app, StreamRequest(action="push", session_id=sid, data=""))
    assert empty_res["status"] == "error" and empty_res["error"]["code"] == "EMPTY_AUDIO"
    analyze_audio_stream(app, StreamRequest(action="cancel", session_id=sid))

    sid2 = analyze_audio_stream(
        app, StreamRequest(action="start", chunk_format="file", sample_rate_hz=16000)
    )["result"]["session_id"]
    push_res = analyze_audio_stream(app, StreamRequest(action="push", session_id=sid2, path=str(speech_en)))
    assert push_res["status"] == "success"
    analyze_audio_stream(app, StreamRequest(action="cancel", session_id=sid2))

    raw_speech, _ = load(speech_en)
    b64 = base64.b64encode(raw_speech[0].astype(np.float32).tobytes()).decode()
    w_res = analyze_audio_stream(
        app,
        StreamRequest(action="start", chunk_format="pcm", sample_rate_hz=16000, channels=1, sample_format="f32le"),
    )["result"]["session_id"]
    sess_f = app.streams._get(w_res)
    with sess_f.buffer.open("wb") as fh:
        fh.write(np.zeros(35 * 16000, dtype=np.float32).tobytes())
    sess_f.samples = 35 * 16000
    p_out = analyze_audio_stream(app, StreamRequest(action="push", session_id=w_res, data=b64))["result"]
    tw_segs = p_out["transcript_window"]["segments"]
    assert tw_segs and all(w["start"] >= p_out["transcript_window"]["start"] for s in tw_segs for w in s["words"])
    analyze_audio_stream(app, StreamRequest(action="cancel", session_id=w_res))

    app.settings.redact_outputs = True
    app.settings.redact_extra_patterns = ["friday"]
    sid4 = analyze_audio_stream(
        app, StreamRequest(action="start", chunk_format="file")
    )["result"]["session_id"]
    analyze_audio_stream(app, StreamRequest(action="push", session_id=sid4, path=str(speech_en)))
    fin = analyze_audio_stream(app, StreamRequest(action="finish", session_id=sid4, tasks=["transcription"]))
    assert fin["status"] == "success"
    assert fin["result"]["reconciliation"]["transcript_word_difference_rate"] == 0.0


def test_bug07_export_fails_cleanly_when_required_stage_unavailable(app, speech_en):
    """BUG-07: export_audio_analysis for srt/vtt/rttm must report error when required backend is unavailable."""
    from audio_mcp.backends.base import UnavailableBackend

    aid = extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["audio_id"]
    app.backends.asr = UnavailableBackend("faster-whisper", "faster_whisper not installed")
    r_srt = export_audio_analysis(app, ExportRequest(audio_id=aid, format="srt", async_mode="never"))
    assert r_srt["status"] == "error"

    app.backends.speaker = UnavailableBackend("ecapa", "speechbrain not installed")
    r_rttm = export_audio_analysis(app, ExportRequest(audio_id=aid, format="rttm", async_mode="never"))
    assert r_rttm["status"] == "error"
    err_code = (r_rttm.get("error") or (r_rttm.get("errors") or [{}])[0]).get("code")
    assert err_code == "MODEL_UNAVAILABLE"


def test_bug08_analyze_audio_reports_partial_when_requested_task_skipped(app, speech_en):
    """BUG-08: analyze_audio with explicit tasks must not return status='success' when all requested tasks skipped."""
    from audio_mcp.backends.base import UnavailableBackend

    app.backends.asr = UnavailableBackend("faster-whisper", "faster_whisper not installed")
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["transcription"], async_mode="never"))
    assert r["status"] in ("partial", "error")
    assert r["errors"] and r["errors"][0]["code"] == "MODEL_UNAVAILABLE"


def test_bug09_url_port_validation_teredo_and_ipv4_mapped_loopback(settings):
    """BUG-09: Invalid URL port is INVALID_INPUT, Teredo loopback is blocked, and ::ffff:127.0.0.1 is loopback."""
    settings.allow_url_fetch = True
    with pytest.raises(AudioMCPError) as ei:
        materialize_source(UrlSource(type="url", url="http://example.org:99999/clip.wav"), settings)
    assert ei.value.code == ErrorCode.INVALID_INPUT and ei.value.retryable is False

    assert _blocked("2001:0000:4136:e378:8000:63bf:80ff:fffe") is True
    assert _blocked("64:ff9b:1::7f00:1") is True
    assert is_loopback("http://[::ffff:127.0.0.1]:11434/v1") is True
    assert is_loopback("http://localhost.:11434/v1") is True


def test_bug10_server_run_resets_job_context(monkeypatch, tmp_path, speech_en):
    """BUG-10: server._run must reset JOB_CONTEXT after completion so pooled worker threads do not leak."""
    from audio_mcp import server

    monkeypatch.setenv("AUDIO_MCP_BACKEND_PROFILE", "fake")
    monkeypatch.setenv("AUDIO_MCP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AUDIO_MCP_ALLOWED_ROOTS", f'["{speech_en.parent.as_posix()}"]')

    async def check():
        async with server.lifespan(None):
            monkeypatch.setattr(server, "_progress_reporter", lambda: (lambda d, t: None))
            # Run synchronous call in current thread via anyio.to_thread.run_sync stub to check leak on same thread
            async def run_inline(fn):
                return fn()

            monkeypatch.setattr(anyio.to_thread, "run_sync", run_inline)
            await server.extract_audio_features(source={"type": "file_path", "path": str(speech_en)})
            assert JOB_CONTEXT.get() is None

    asyncio.run(check())


def test_bug11_llm_parse_with_surrounding_braces_and_config_regex_validation():
    """BUG-11: _parse handles prose braces around JSON; Settings rejects invalid regex; _pid_alive(0) is False."""
    out = 'Here is the analysis for {seg_0001}:\n{"answer": "Friday", "claims": [], "insufficient": false}\nDone {ok}.'
    parsed = _parse(out, ("answer",))
    assert parsed["answer"] == "Friday"

    with pytest.raises(Exception):
        Settings(redact_extra_patterns=["[unclosed_group"])

    assert _pid_alive(0) is False
    assert _pid_alive(-5) is False


def test_review_followups(app, dialogue, tmp_path):
    """PR review: NAT64 local-use, FTS marks + migration, NFC/CJK tokens, echoed-evidence parse, explicit tasks."""
    assert _blocked("64:ff9b:1::808:808")  # RFC 8215 local-use prefix embedding public 8.8.8.8
    assert not _blocked("64:ff9b::808:808")  # RFC 6052 prefix still unwraps to the (public) IPv4

    assert _tokens("\u0915\u093e") != _tokens("\u0915\u093f")
    assert _tokens("e\u0301cole") == _tokens("\u00e9cole")  # NFC: same word either way
    assert _tokens("\u732b")  # single CJK character survives

    with pytest.raises(ValueError):  # echoed evidence object is not a response
        _parse('{"id": "seg_0001", "start": 0.0, "text": "hi"} then {"answer": "x"} ', ("sentences",))
    assert _parse('{"id": "seg_0001", "start": 0} {"answer": "x"}', ("answer",)) == {"answer": "x"}

    # an index built with the default tokenizer is rebuilt, and kept marks distinguish ka / ki
    import sqlite3
    db = tmp_path / "lib.db"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE segments(rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, audio_id TEXT, kind TEXT,
      start REAL, end REAL, text TEXT, label TEXT, speaker_id TEXT, embedding BLOB);
      CREATE VIRTUAL TABLE segments_fts USING fts5(text, content='segments', content_rowid='rowid');
      INSERT INTO segments(id,audio_id,kind,text) VALUES ('a','x','speech','\u0915\u093e'), ('b','x','speech','\u0915\u093f');
      INSERT INTO segments_fts(rowid,text) VALUES (1,'\u0915\u093e'), (2,'\u0915\u093f');""")
    con.commit(); con.close()
    lib = Library(db)
    assert list(lib._fts("\u0915\u093e")) == [1]
    lib.close()

    r = analyze_audio(app, AnalyzeRequest(source=fp(dialogue), tasks=["transcription", "speaker_diarization"],
                                          options={"identify_speakers": False}))
    assert "speakers" not in r["analysis"]
    assert "speaker_assign" not in r["processing"]["stages_run"]


def test_uncertainty_cache_key_tracks_quality_rules(app):
    """PR review: uncertainty reads quality, so its key includes quality's config only when quality ran."""
    cfg = STAGES["uncertainty"].cfg
    base = cfg(app, {})
    assert cfg(app, {"_stages": ["quality"]}) != cfg(app, {"_stages": ["events"]}) != base
    assert "quality" not in cfg(app, {"_stages": ["events"]})

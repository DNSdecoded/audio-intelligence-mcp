import pytest

from audio_mcp.backends.base import UnavailableBackend
from audio_mcp.context import build_app_context
from audio_mcp.ingest.sources import FilePathSource
from audio_mcp.pipeline.runner import run_analysis


@pytest.fixture
def app(settings):
    return build_app_context(settings)


def src(p):
    return FilePathSource(type="file_path", path=str(p))


def test_runs_with_deps_and_caches(app, speech_en):
    o1 = run_analysis(app, src(speech_en), ["transcription"], {})
    assert o1.processing.stages_run == ["vad", "transcription"] and o1.results["transcription"]["segments"]
    o2 = run_analysis(app, src(speech_en), ["transcription"], {})
    assert o2.processing.cache_hits == ["vad", "transcription"] and o2.processing.stages_run == []


def test_failed_stage_reports_and_skips_dependents(app, speech_en, monkeypatch):
    monkeypatch.setattr(app.backends.asr, "transcribe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    o = run_analysis(app, src(speech_en), ["transcription", "language", "features"], {})
    assert o.errors[0]["stage"] == "transcription" and "features" in o.results
    assert {"stage": "language", "reason": "dependency transcription failed"} in o.processing.stages_skipped


def test_unavailable_backend_skipped(app, speech_en):
    app.backends.asr = UnavailableBackend("faster-whisper", "faster_whisper not installed")
    o = run_analysis(app, src(speech_en), ["transcription"], {})
    assert o.processing.stages_skipped[0]["reason"].startswith("faster_whisper not installed")


def test_unregistered_stage_reported(app, speech_en):
    o = run_analysis(app, src(speech_en), ["summary"], {})
    assert {"stage": "summary", "reason": "not implemented in this version"} in o.processing.stages_skipped


@pytest.mark.parametrize("name", ["short", "silence"])
def test_tiny_and_silent_inputs(app, synth, name):
    o = run_analysis(app, src(synth[name]), ["transcription", "events", "features"], {})
    assert o.errors == [] and o.results["transcription"]["segments"] == []

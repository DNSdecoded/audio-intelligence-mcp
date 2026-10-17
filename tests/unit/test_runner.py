import pytest

from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.pipeline.stages import STAGES, Stage

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
    o = run_analysis(app, src(speech_en), ["not_a_stage"], {})
    assert {"stage": "not_a_stage", "reason": "not implemented in this version"} in o.processing.stages_skipped


@pytest.mark.parametrize("name", ["short", "silence"])
def test_tiny_and_silent_inputs(app, synth, name):
    o = run_analysis(app, src(synth[name]), ["transcription", "events", "features"], {})
    assert o.errors == [] and o.results["transcription"]["segments"] == []


def _raise(e):
    def f(*a, **k):
        raise e
    return f


def test_model_load_failure_is_skip_not_error(app, speech_en, monkeypatch):
    monkeypatch.setattr(app.backends.asr, "transcribe", _raise(AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, "boom load")))
    o = run_analysis(app, src(speech_en), ["transcription", "language"], {})
    assert o.errors == []
    assert {"stage": "transcription", "reason": "boom load"} in o.processing.stages_skipped
    assert {"stage": "language", "reason": "dependency transcription skipped"} in o.processing.stages_skipped


def test_vad_runtime_failure_falls_back(app, speech_en):
    class Bad:
        info = ModelInfo(stage="vad", name="bad-vad", version="1")
        available = staticmethod(lambda: (True, None))
        speech_regions = staticmethod(_raise(RuntimeError("onnx")))

    app.backends.vad = Bad()
    o = run_analysis(app, src(speech_en), ["vad"], {})
    assert o.errors == [] and o.results["vad"]["backend"] == "energy-vad"
    assert any(w["code"] == "VAD_FALLBACK" and "RuntimeError" in w["message"] for w in o.warnings)
    assert [m.name for m in o.processing.models_used] == ["energy-vad"]


def test_soft_dep_failure_does_not_skip_dependent(app, speech_en, monkeypatch):
    seen = {}

    def fn(ctx):
        seen["keys"] = set(ctx.results)
        return {"ok": True}

    monkeypatch.setitem(STAGES, "softy", Stage("softy", ("metadata",), fn, lambda a, o: {}, soft_deps=("transcription",)))
    monkeypatch.setattr(app.backends.asr, "transcribe", _raise(RuntimeError("x")))
    o = run_analysis(app, src(speech_en), ["softy"], {})
    assert o.results["softy"] == {"ok": True} and "transcription" not in seen["keys"]
    assert o.errors[0]["stage"] == "transcription"


def test_cancel_before_run(app, speech_en):
    import threading

    ev = threading.Event()
    ev.set()
    with pytest.raises(AudioMCPError) as ei:
        run_analysis(app, src(speech_en), ["transcription"], {}, cancel_event=ev)
    assert ei.value.code == ErrorCode.CANCELLED


def test_progress_ends_at_total(app, speech_en):
    calls = []
    run_analysis(app, src(speech_en), ["transcription", "language"], {}, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_late_stage_not_cached_after_failed_dep(app, speech_en, monkeypatch):  # I1
    with monkeypatch.context() as m:
        m.setattr(app.backends.asr, "transcribe", _raise(RuntimeError("flaky")))
        o1 = run_analysis(app, src(speech_en), ["timeline"], {})
    assert o1.errors and not any(n["type"] == "speech_segment" for n in o1.results["timeline"]["nodes"])
    o2 = run_analysis(app, src(speech_en), ["timeline"], {})
    assert "timeline" in o2.processing.stages_run and "timeline" not in o2.processing.cache_hits
    assert any(n["type"] == "speech_segment" for n in o2.results["timeline"]["nodes"])


def test_vad_fallback_not_cached(app, speech_en):  # I1
    real = app.backends.vad
    calls = []

    class Flaky:
        info = ModelInfo(stage="vad", name="flaky-vad", version="1")
        available = staticmethod(lambda: (True, None))

        @staticmethod
        def speech_regions(x):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("onnx")
            return real.speech_regions(x)

    app.backends.vad = Flaky()
    run_analysis(app, src(speech_en), ["features"], {})
    o = run_analysis(app, src(speech_en), ["features"], {})
    assert o.processing.cache_hits == [] and o.processing.stages_run == ["vad", "features"]
    assert o.results["vad"]["backend"] == "flaky-vad" and len(calls) == 2

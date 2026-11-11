import numpy as np
import pytest

from audio_mcp.backends.base import ASRResult
from audio_mcp.ingest.sources import FilePathSource
from audio_mcp.pipeline.stages import STAGES
from audio_mcp.tools import handlers
from audio_mcp.tools.schemas import TranscribeRequest
from audio_mcp.context import build_app_context


def _app(settings, **over):
    for k, v in over.items():
        setattr(settings, k, v)
    app = build_app_context(settings)
    seen = []
    real = app.backends.asr.transcribe

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    app.backends.asr.transcribe = spy
    return app, seen


def _req(path, **kw):
    return TranscribeRequest(source=FilePathSource(type="file_path", path=str(path)), async_mode="never", **kw)


def test_hotwords_forwarded_and_keyed(settings, speech_en):
    app, seen = _app(settings)
    handlers.transcribe_audio(app, _req(speech_en, hotwords=["Kubernetes", "Sakhinala"]))
    assert seen[-1]["hotwords"] == "Kubernetes Sakhinala"
    n = len(seen)
    handlers.transcribe_audio(app, _req(speech_en, hotwords=["Kubernetes", "Sakhinala"]))
    assert len(seen) == n  # same hotwords: cached transcript reused
    handlers.transcribe_audio(app, _req(speech_en, hotwords=["Other"]))
    assert len(seen) == n + 1 and seen[-1]["hotwords"] == "Other"  # new hotwords: recomputed
    handlers.transcribe_audio(app, _req(speech_en))
    assert seen[-1]["hotwords"] is None


def test_hotwords_reach_faster_whisper_call():
    from audio_mcp.backends.whisper import FasterWhisperASR

    class Model:
        kw = None

        def transcribe(self, x, **kw):
            Model.kw = kw
            return iter(()), type("I", (), {"language": "en", "language_probability": 1.0})()

    asr = FasterWhisperASR(type("S", (), {"asr_model": "small"})())
    asr._model = Model()
    asr.transcribe(np.zeros(16000, dtype=np.float32), hotwords="Kubernetes")
    assert Model.kw["hotwords"] == "Kubernetes"


@pytest.mark.parametrize("lang,model,warned", [
    ("te", "small", True), ("ar", "base", True), ("hi", "tiny", True),
    ("te", "large-v3", False), ("en", "small", False)])
def test_low_resource_language_warning(settings, speech_en, lang, model, warned):
    app, _ = _app(settings, asr_model=model)
    real = app.backends.asr.transcribe

    def forced(*a, **kw):
        r = real(*a, **kw)
        return ASRResult(r.segments, lang, 0.9, r.languages, r.dropped_low_speech)

    app.backends.asr.transcribe = forced
    out = handlers.transcribe_audio(app, _req(speech_en))
    codes = [w["code"] for w in out["warnings"]]
    assert ("LOW_RESOURCE_LANGUAGE" in codes) is warned

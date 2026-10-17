import base64
from pathlib import Path

import pytest

from audio_mcp.analysis.events import EVENT_CATEGORIES
from audio_mcp.context import build_app_context
from audio_mcp.tools.handlers import analyze_audio, detect_audio_events, extract_audio_features, transcribe_audio
from audio_mcp.tools.schemas import AnalyzeRequest, DetectEventsRequest, FeaturesRequest, TranscribeRequest
from pydantic import ValidationError


@pytest.fixture
def app(settings):
    return build_app_context(settings)


def fp(p):
    return {"type": "file_path", "path": str(p)}


def test_analyze_shape(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["transcription", "language_detection", "sound_event_detection", "acoustic_features"]))
    a = r["analysis"]
    assert r["status"] == "success" and r["audio_id"].startswith("aud_")
    assert {"audio_metadata", "language", "transcription", "sound_events", "acoustic_features"} <= a.keys()
    assert a["audio_metadata"]["sample_rate_hz"] == 16000 and r["processing"]["models_used"]


def test_bad_path_error_shape(app):
    r = transcribe_audio(app, TranscribeRequest(source={"type": "file_path", "path": str(Path.home().parent)}))
    assert r["status"] == "error" and r["error"]["code"] in {"PATH_NOT_ALLOWED", "INVALID_INPUT"} and r["error"]["retryable"] is False


def test_unsupported_format(app, synth):
    r = extract_audio_features(app, FeaturesRequest(source=fp(synth["not_audio"])))
    assert r["error"]["code"] in {"CORRUPT_AUDIO", "UNSUPPORTED_AUDIO_FORMAT"}


def test_events_categories(app, synth):
    r = detect_audio_events(app, DetectEventsRequest(source=fp(synth["alarm"]), categories=["alarm"]))
    assert r["status"] == "success" and r["result"]["events"] and all(e["label"] in EVENT_CATEGORIES["alarm"] for e in r["result"]["events"])


def test_emotions_unsupported_warning(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["acoustic_features"], options={"detect_emotions": True}))
    assert any(w["code"] == "UNSUPPORTED_FEATURE" for w in r["warnings"])


def test_cache_reuse_across_source_types(app, speech_en):  # Review Focus #5
    transcribe_audio(app, TranscribeRequest(source=fp(speech_en)))
    r = transcribe_audio(app, TranscribeRequest(source={"type": "base64", "data": base64.b64encode(speech_en.read_bytes()).decode()}))
    assert "transcription" in r["processing"]["cache_hits"]


def test_options_strip_confidence_and_words(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["transcription", "language_detection"],
                                          options={"include_confidence": False, "word_level_timestamps": False}))
    s = repr(r["analysis"])
    assert "'confidence'" not in s and "'words'" not in s


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        TranscribeRequest(source=fp("x"), bogus=1)

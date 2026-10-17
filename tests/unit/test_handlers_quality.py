from pathlib import Path

from conftest import fp
from audio_mcp.tools.handlers import analyze_audio, inspect_audio_quality
from audio_mcp.tools.schemas import AnalyzeRequest, InspectQualityRequest


def test_handler_preserves_original(app, synth):
    p = synth["low_level"]; before = p.read_bytes()
    r = inspect_audio_quality(app, InspectQualityRequest(source=fp(p), apply_preprocessing=True))["result"]["preprocessing"]
    assert p.read_bytes() == before and Path(r["variant_path"]).exists() and r["after"]["rms_dbfs"] > r["before"]["rms_dbfs"]
    assert [t["op"] for t in r["transforms"]] == ["normalize"]


def test_report_only(app, synth):
    r = inspect_audio_quality(app, InspectQualityRequest(source=fp(synth["clipped"])))
    assert r["status"] == "success" and "preprocessing" not in r["result"]
    assert any(i["code"] == "CLIPPING" for i in r["result"]["issues"])


def test_compare_transcription(app, speech_en):
    r = inspect_audio_quality(app, InspectQualityRequest(source=fp(speech_en), apply_preprocessing=["normalize"],
                                                         compare_transcription=True))["result"]["preprocessing"]
    c = r["transcription_comparison"]
    assert {"before", "after", "note"} <= c.keys() and "mean_word_confidence" in c["before"]


def test_analyze_maps_quality(app, synth):
    r = analyze_audio(app, AnalyzeRequest(source=fp(synth["clipped"]), tasks=["audio_quality"]))
    assert r["analysis"]["audio_quality"]["overall"] == "poor"

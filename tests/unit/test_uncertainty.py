import pytest

from conftest import fp
from audio_mcp.analysis.uncertainty import find_uncertainties
from audio_mcp.eval.metrics import word_error_rate
from audio_mcp.ingest.sources import FilePathSource
from audio_mcp.pipeline.runner import run_analysis
from audio_mcp.tools.handlers import analyze_audio
from audio_mcp.tools.schemas import AnalyzeRequest


def w(s, e, t, c):
    return {"start": s, "end": e, "text": t, "confidence": c, "confidence_kind": "model_probability"}


def seg(i, s, e, text, conf=None, words=()):
    d = {"id": i, "start": s, "end": e, "text": text, "words": list(words)}
    return {**d, "confidence": conf, "confidence_kind": "model_probability"} if conf is not None else d


def test_flags():
    r = {"transcription": {"segments": [seg("seg_0001", 0, 2, "a b c", conf=.3, words=[w(0, .5, "a", .2), w(.5, 1, "b", .3), w(1, 2, "c", .9)]),
                                        seg("seg_0002", 5, 6, "x", conf=.9)]},
         "vad": {"speech_regions": [{"start": 0, "end": 2}, {"start": 8, "end": 10}]}}
    out = find_uncertainties(r, low_conf=.5)
    types = {i["type"] for i in out["items"]}
    assert {"LOW_CONFIDENCE_SEGMENT", "LOW_CONFIDENCE_WORDS", "ASR_WITHOUT_VAD", "VAD_WITHOUT_ASR"} <= types
    assert out["summary"]["low_confidence_word_ratio"] == pytest.approx(2 / 3, abs=1e-3)
    words = next(i for i in out["items"] if i["type"] == "LOW_CONFIDENCE_WORDS")
    assert (words["start"], words["end"], words["evidence_ids"]) == (0, 1, ["seg_0001"])


def test_tagger_and_quality_items():
    r = {"transcription": {"segments": []}, "vad": {"speech_regions": [{"start": 0, "end": 3}]},
         "events": {"events": [{"id": "evt_0001", "label": "Speech", "start": 5, "end": 7}]},
         "quality": {"issues": [{"code": "CLIPPING", "severity": "high", "message": "m"},
                                {"code": "ECHO", "severity": "medium", "message": "m"}]}}
    types = [i["type"] for i in find_uncertainties(r, low_conf=.5)["items"]]
    assert types.count("TAGGER_VAD_DISAGREEMENT") == 2 and types.count("QUALITY_AFFECTED") == 1


def test_wer():
    assert word_error_rate("a b c d", "a x c") == pytest.approx(0.5)
    assert word_error_rate("Hello, world!", "hello world") == 0 and word_error_rate("", "") == 0


def test_thorough_cross_check(app, speech_en):
    o = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["uncertainty"], {"budget": "thorough"})
    cc = o.results["uncertainty"]["summary"]["asr_cross_check"]
    assert cc["word_disagreement_rate"] >= 0 and "method" in cc
    o2 = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["uncertainty"], {})
    assert o2.results["uncertainty"]["summary"]["asr_cross_check"] is None


def test_analyze_maps_uncertainty(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["uncertainty"]))
    assert "items" in r["analysis"]["uncertainty"]

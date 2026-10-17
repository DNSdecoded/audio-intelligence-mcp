import pytest

from conftest import load
from audio_mcp.dsp.features import extract_features
from audio_mcp.dsp.quality import diagnose_quality


def _q(path):
    x, sr = load(path)
    return diagnose_quality(x, sr, extract_features(x, sr)["values"], [])


@pytest.mark.parametrize("name,code", [("clipped", "CLIPPING"), ("low_level", "LOW_LEVEL"), ("silence", "SILENCE"),
                                       ("hum", "HUM"), ("dc_offset", "DC_OFFSET"), ("noise", "HIGH_NOISE")])
def test_issue_detected(synth, name, code):
    assert code in {i["code"] for i in _q(synth[name])["issues"]}


def test_clean_sine(synth):
    q = _q(synth["sine"])
    assert not {"CLIPPING", "LOW_LEVEL", "DC_OFFSET", "HIGH_NOISE"} & {i["code"] for i in q["issues"]}


def test_shape_and_severity(synth):
    q = _q(synth["clipped"])
    clip = next(i for i in q["issues"] if i["code"] == "CLIPPING")
    assert clip["severity"] == "high" and "not recoverable" in clip["message"]
    assert q["overall"] == "poor" and {"metrics", "issues", "recommendations", "overall"} <= q.keys()
    multi = next(i for i in _q(synth["sine"])["issues"] if i["code"] == "MULTICHANNEL")
    assert multi["severity"] == "low"


def test_recommendations_dedup(synth):
    q = _q(synth["low_level"])
    assert q["recommendations"].count("normalize") == 1 and "normalize" in q["recommendations"]

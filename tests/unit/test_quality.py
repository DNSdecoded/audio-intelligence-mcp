import numpy as np
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


def _codes(x, sr):
    return {i["code"] for i in diagnose_quality(x, sr, extract_features(x, sr)["values"], [])["issues"]}


@pytest.fixture
def speech():
    from conftest import FIXTURES_DIR

    x, sr = load(FIXTURES_DIR / "speech_en.wav")
    return np.atleast_2d(x)[0], sr


def test_clipping_below_full_scale(speech):
    m, sr = speech
    assert "CLIPPING" in _codes(np.clip(m / np.abs(m).max() * 4, -1, 1) * 0.7, sr)  # clipped, then turned down
    assert "CLIPPING" not in _codes(m, sr)


def test_echo_only_with_discrete_echo(speech):
    m, sr = speech
    d = int(0.12 * sr)
    e = m.copy()
    e[d:] += 0.6 * m[:-d]
    assert "ECHO" in _codes(e, sr) and "ECHO" not in _codes(m, sr)


def test_bandwidth_limited_by_lowpass(speech):
    from scipy.signal import butter, sosfilt

    m, sr = speech
    assert "BANDWIDTH_LIMITED" in _codes(sosfilt(butter(8, 3400, fs=sr, output="sos"), m), sr)
    assert "BANDWIDTH_LIMITED" not in _codes(m, sr)


def test_level_ignores_long_silence(speech):
    m, sr = speech
    c = _codes(np.concatenate([m, np.zeros(3 * len(m), dtype=m.dtype)]), sr)
    assert "SILENCE" in c and "LOW_LEVEL" not in c

import numpy as np
import pytest
from pydantic import ValidationError

from conftest import fp
from audio_mcp.compare.compare import baseline_anomalies, compare_pair, window_features
from audio_mcp.tools.handlers import compare_audio
from audio_mcp.tools.schemas import CompareRequest


def test_window_features_shape():
    x = np.zeros((1, 16000 * 3), dtype="float32")
    feats, names = window_features(x, 16000)
    assert feats.shape == (3, len(names)) and names[0] == "rms_dbfs" and feats[0, 0] == -100
    assert len([n for n in names if n.startswith("band_")]) == 7


def test_pair_level_delta(app, synth):
    r = compare_audio(app, CompareRequest(sources=[fp(synth["sine"]), fp(synth["low_level"])]))
    c = r["result"]["comparisons"][0]
    d = c["measured_differences"]["features"]["rms_dbfs"]
    assert d["unit"] == "dBFS" and d["delta"] < -30
    assert any(i["kind"] == "interpretation" and "louder" in i["statement"] for i in c["possible_interpretations"])
    assert "audio_id" not in r and len(r["result"]["audio_ids"]) == 2
    assert c["similarity"]["audio_embedding_cosine"] <= 1.0


def test_events_and_transcript_diff():
    a = {"audio_id": "a", "duration": 60.0, "features": {"units": {}, "values": {}},
         "events": [{"label": "Speech", "start": 0, "end": 5}], "transcript": "the pump is quiet", "embedding": None}
    b = {**a, "audio_id": "b", "events": [{"label": "Speech", "start": 0, "end": 5}, {"label": "Alarm", "start": 9, "end": 10}],
         "transcript": "the pump is loud"}
    c = compare_pair(a, b)
    ev = c["measured_differences"]["events"]
    assert ev["only_in_b"] == ["Alarm"] and ev["only_in_a"] == []
    assert c["measured_differences"]["transcript"]["b_only_words"] == ["loud"]
    assert c["similarity"] == {} and any("Alarm" in i["statement"] for i in c["possible_interpretations"])


def test_baseline_localizes_anomaly():
    base = (np.sin(2 * np.pi * 220 * np.arange(16000 * 5) / 16000) * .3).astype("float32")[None]
    tgt = base.copy()
    tgt[0, 32000:48000] += np.random.default_rng(0).normal(0, .3, 16000).astype("float32")
    out = baseline_anomalies(window_features(tgt, 16000), [window_features(base, 16000)])
    w = out["anomalous_windows"]
    assert len(w) == 1 and w[0]["start"] == pytest.approx(2.0, abs=0.01) and w[0]["end"] == pytest.approx(3.0, abs=0.01)
    assert out["baseline_windows"] == 5


def test_requires_two_without_baseline():
    with pytest.raises(ValidationError):
        CompareRequest(sources=[{"type": "file_path", "path": "x.wav"}])


def test_baseline_mode(app, synth):
    r = compare_audio(app, CompareRequest(sources=[fp(synth["noise"])], baseline=[fp(synth["sine"])]))["result"]
    assert next(iter(r["baseline"].values()))["anomalous_features"]


def test_aspects_filter(app, synth):
    c = compare_audio(app, CompareRequest(sources=[fp(synth["sine"]), fp(synth["noise"])],
                                          aspects=["features"]))["result"]["comparisons"][0]
    assert set(c["measured_differences"]) == {"features", "bands"}

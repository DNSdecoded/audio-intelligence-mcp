import pytest
import numpy as np

from audio_mcp.dsp.features import extract_features
from audio_mcp.types import Interval
from conftest import load


def test_sine_level(synth):
    v = extract_features(*load(synth["sine"]))["values"]
    assert abs(v["rms"] - 0.5 / np.sqrt(2)) < 1e-3 and abs(v["peak_amplitude"] - 0.5) < 1e-3
    assert abs(v["crest_factor_db"] - 3.01) < 0.05 and v["clipping_ratio"] == 0
    assert abs(v["spectral_centroid_hz"] - 440) < 30 and abs(v["f0_median_hz"] - 440) < 5


def test_clipped(synth):
    assert extract_features(*load(synth["clipped"]), ["level"])["values"]["clipping_ratio"] > 0.5


def test_silence(synth):
    v = extract_features(*load(synth["silence"]))["values"]
    assert v["silence_ratio"] == 1.0 and v["rms_dbfs"] is None and v["f0_mean_hz"] is None


def test_bands_sum_and_units(synth):
    f = extract_features(*load(synth["noise"]), ["bands"])
    assert abs(sum(f["values"]["band_energy"].values()) - 1) < 1e-6 and f["units"]["band_energy"] == "ratio"


def test_beeps_onsets(synth):
    v = extract_features(*load(synth["alarm"]), ["rhythm"])["values"]
    assert 4 <= v["onset_rate_per_s"] * 4.0 <= 10


def test_snr_vad_and_json(synth):
    import json

    x, sr = load(synth["alarm"])
    v = extract_features(x, sr, speech_regions=[Interval(start=1.0, end=3.0)])["values"]
    assert v["snr_method"] == "vad" and v["snr_db_estimate"] > 20
    json.dumps(v, allow_nan=False)


def test_blocked_matches_default(synth, monkeypatch):
    from audio_mcp.dsp import features

    for k in ("alarm", "sine"):
        x, sr = load(synth[k])
        full = extract_features(x, sr)["values"]
        monkeypatch.setattr(features, "_BLOCK_FRAMES", 7)
        got = extract_features(x, sr)["values"]
        assert got.keys() == full.keys()
        for key, want in full.items():
            if isinstance(want, dict):
                want, g = list(want.values()), list(got[key].values())
            else:
                g = got[key]
            assert g == pytest.approx(want, abs=1e-6) if want is not None else g is None
        monkeypatch.undo()


def test_bands_clipped_to_nyquist(synth):
    b = extract_features(*load(synth["multichannel"]), ["bands"])["values"]["band_energy"]
    assert list(b) == ["0-250", "250-500", "500-1000", "1000-2000", "2000-4000"]
    assert abs(sum(b.values()) - 1) < 1e-6

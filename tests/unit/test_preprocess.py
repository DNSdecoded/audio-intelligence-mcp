import numpy as np
import pytest

from conftest import load
from audio_mcp.dsp.preprocess import PREPROCESS_OPS, apply_preprocessing


def test_normalize_and_dc(synth):
    x, sr = load(synth["dc_offset"])
    y, _, t = apply_preprocessing(x, sr, ["normalize", "remove_dc"])
    assert abs(y.mean()) < 1e-3 and abs(20 * np.log10(np.sqrt((y ** 2).mean())) + 20) < 0.5
    assert [s["op"] for s in t] == ["remove_dc", "normalize"]


def test_mono_resample_trim(synth):
    x, sr = load(synth["alarm"])
    y, sr2, t = apply_preprocessing(x, sr, ["trim_silence", "resample_16k", "mono"])
    assert sr2 == 16000 and y.shape[0] == 1
    trim = next(s for s in t if s["op"] == "trim_silence")
    assert trim["params"]["offset_seconds"] == pytest.approx(1.0, abs=0.05)


def test_highpass_removes_hum(synth):
    x, sr = load(synth["hum"])
    y, _, _ = apply_preprocessing(x, sr, ["highpass_80hz"])
    assert np.sqrt((y ** 2).mean()) < 0.3 * np.sqrt((x ** 2).mean())


def test_denoise_reduces_noise_floor(synth):
    x, sr = load(synth["noise"])
    y, _, _ = apply_preprocessing(x, sr, ["denoise"])
    assert y.shape == x.shape and np.sqrt((y ** 2).mean()) < np.sqrt((x ** 2).mean())


def test_unknown_op_rejected(synth):
    x, sr = load(synth["sine"])
    with pytest.raises(ValueError):
        apply_preprocessing(x, sr, ["boost"])
    assert PREPROCESS_OPS[0] == "remove_dc" and PREPROCESS_OPS[-1] == "normalize"

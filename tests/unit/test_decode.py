import shutil

import numpy as np
import pytest

from audio_mcp.envelope import AudioMCPError
from audio_mcp.ingest.decode import decode, probe


@pytest.mark.parametrize("ext", ["flac", "mp3", "ogg", "m4a", "webm", "aiff"])
def test_formats_decode(synth, settings, ext):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg absent")
    p = synth[f"speech_en.{ext}"]
    info = probe(p, settings)
    d = decode(p, info, settings)
    assert d.mono16k.dtype == np.float32 and abs(len(d.mono16k) / 16000 - info.duration_seconds) < 0.1


def test_odd_layouts(synth, settings):  # Review Focus #4
    i = probe(synth["multichannel"], settings)
    d = decode(synth["multichannel"], i, settings)
    assert (i.channels, i.sample_rate_hz) == (6, 8000) and d.native.shape[0] == 6 and d.mono16k.ndim == 1
    h = probe(synth["hires"], settings)
    assert h.sample_rate_hz == 96000 and h.bit_depth == 24


@pytest.mark.parametrize("name,code", [("corrupt", "CORRUPT_AUDIO"), ("not_audio", "CORRUPT_AUDIO"), ("empty", "EMPTY_AUDIO")])
def test_bad_inputs(synth, settings, name, code):
    with pytest.raises(AudioMCPError) as e:
        decode(synth[name], probe(synth[name], settings), settings)
    assert e.value.code == code


def test_duration_limit(synth, settings):
    settings.max_duration_seconds = 1.0
    with pytest.raises(AudioMCPError) as e:
        probe(synth["sine"], settings)
    assert e.value.code == "DURATION_TOO_LONG"


def test_partial_decode_warns(synth, settings, tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg absent")
    b = synth["speech_en.mp3"].read_bytes()
    p = tmp_path / "partial.mp3"
    p.write_bytes(b[: int(len(b) * 0.9)] + b"garbage")
    d = decode(p, probe(p, settings), settings)
    assert isinstance(d.decode_warnings, list) and len(d.mono16k) > 0


def test_timeout_code(synth, settings, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(AudioMCPError) as e:
        probe(synth["sine"], settings)
    assert e.value.code == "TIMEOUT" and e.value.retryable

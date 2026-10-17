"""Deterministic synthetic audio fixtures. Pure numpy/soundfile; ffmpeg only for format variants."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SPEECH_EN = Path(__file__).parent / "audio" / "speech_en.wav"


def _t(dur, sr):
    return np.arange(int(round(dur * sr))) / sr


def _sine(freq, amp, dur, sr):
    return amp * np.sin(2 * np.pi * freq * _t(dur, sr))


def _noise(rng, amp, dur, sr):
    return rng.uniform(-amp, amp, int(round(dur * sr)))


def _w(path, data, sr, subtype="PCM_16"):
    sf.write(path, np.clip(data, -1.0, 1.0), sr, subtype=subtype)
    return path


def _burst(rng, sr, peak, dur=0.05):
    n = int(dur * sr)
    return peak * rng.uniform(-1, 1, n) * np.exp(-np.arange(n) / (n / 5))


def make_synthetic(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    p: dict[str, Path] = {}

    s = _sine(440, 0.5, 2.0, 44100)
    p["sine"] = _w(out_dir / "sine.wav", np.stack([s, s], 1), 44100)
    p["silence"] = _w(out_dir / "silence.wav", np.zeros(3 * 16000), 16000)
    p["noise"] = _w(out_dir / "noise.wav", _noise(rng, 0.1, 3.0, 16000), 16000)
    p["clipped"] = _w(out_dir / "clipped.wav", _sine(440, 2.0, 2.0, 44100), 44100)

    sr = 32000
    a = _noise(rng, 0.001, 4.0, sr)
    tt = _t(2.0, sr)
    on = (tt % 0.3) < 0.2
    a[sr:3 * sr] = 0.5 * np.sin(2 * np.pi * 1000 * tt) * on
    p["alarm"] = _w(out_dir / "alarm.wav", a, sr)

    sr = 16000
    imp = _noise(rng, 0.001, 3.0, sr)
    for at in (0.5, 1.5, 2.5):
        i = int(at * sr)
        b = _burst(rng, sr, 0.8)
        imp[i:i + len(b)] += b
    p["impulses"] = _w(out_dir / "impulses.wav", imp, sr)

    sp = np.concatenate([
        _sine(220, 0.3, 1.0, sr),
        np.zeros(int(0.05 * sr)),
        _sine(220, 0.3, 0.95, sr),
        _noise(rng, 0.05, 1.0, sr),
    ])
    p["spliced"] = _w(out_dir / "spliced.wav", sp, sr)
    p["hum"] = _w(out_dir / "hum.wav", _sine(50, 0.3, 3, sr) + _noise(rng, 0.01, 3, sr), sr)
    p["low_level"] = _w(out_dir / "low_level.wav", _sine(440, 0.005, 2, sr), sr)
    p["dc_offset"] = _w(out_dir / "dc_offset.wav", _sine(440, 0.3, 2, sr) + 0.05, sr)
    mc = _sine(440, 0.3, 1.0, 8000)
    p["multichannel"] = _w(out_dir / "multichannel.wav", np.tile(mc[:, None], (1, 6)), 8000)
    p["hires"] = _w(out_dir / "hires.flac", _sine(1000, 0.5, 1.0, 96000), 96000, "PCM_24")
    p["short"] = _w(out_dir / "short.wav", _sine(440, 0.3, 0.3, sr), sr)

    ffmpeg = shutil.which("ffmpeg")
    head = b"ID3" + bytes(2045)  # fallback when ffmpeg is missing
    if ffmpeg and SPEECH_EN.exists():
        mp3 = out_dir / "_good.mp3"
        _ffmpeg(ffmpeg, SPEECH_EN, mp3)
        head = mp3.read_bytes()[:2048]
        mp3.unlink()
    p["corrupt"] = out_dir / "corrupt.mp3"
    p["corrupt"].write_bytes(head + rng.integers(0, 256, 4096, dtype=np.uint8).tobytes())
    p["not_audio"] = out_dir / "not_audio.wav"
    p["not_audio"].write_text("hello")
    p["empty"] = out_dir / "empty.wav"
    p["empty"].write_bytes(b"")

    if ffmpeg and SPEECH_EN.exists():
        for ext in ("flac", "mp3", "ogg", "m4a", "webm", "aiff"):
            p[f"speech_en.{ext}"] = _ffmpeg(ffmpeg, SPEECH_EN, out_dir / f"speech_en.{ext}")
    return p


def _ffmpeg(ffmpeg, src, dst):
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), str(dst)], check=True)
    return dst


def make_long(out_dir: Path) -> Path:
    """30 min 16 kHz mono FLAC; only for `long`-marked tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "long.flac"
    sr = 16000
    chunk = _sine(220, 0.1, 60, sr)  # 60 s is a whole number of 220 Hz cycles: seamless
    with sf.SoundFile(path, "w", sr, 1, format="FLAC", subtype="PCM_16") as f:
        for _ in range(30):
            f.write(chunk)
    return path

"""Signal-level integrity observations (possible edit points). Never an authenticity verdict."""
import numpy as np

from audio_mcp.dsp.features import _frames, frame_rms

INTEGRITY_LIMITATIONS = (
    "These are signal-level observations that can also arise from normal recording, compression, or editing "
    "workflows. They are not evidence that a recording is authentic, manipulated, or synthetic, and no such "
    "determination is made.")
_SYNTHETIC = {"status": "not_available", "reason": "no local synthetic-speech model configured"}
_MERGE = 0.05


def _obs(type_, start, end, magnitude, unit, description):
    return {"type": type_, "start": round(float(start), 3), "end": round(float(end), 3),
            "magnitude": round(float(magnitude), 3), "unit": unit, "description": description}


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-10))


def _zero_runs(m, sr):
    out, min_len, ctx = [], int(0.005 * sr), int(0.1 * sr)
    z = np.concatenate([[False], m == 0, [False]])
    starts, ends = np.flatnonzero(~z[:-1] & z[1:]), np.flatnonzero(z[:-1] & ~z[1:])
    for s, e in zip(starts, ends):
        if e - s < min_len or s < ctx or e + ctx > len(m):
            continue
        before, after = m[s - ctx:s], m[e:e + ctx]
        if _db(np.sqrt((before ** 2).mean())) > -45 and _db(np.sqrt((after ** 2).mean())) > -45:
            out.append(_obs("ZERO_RUN", s / sr, e / sr, (e - s) / sr * 1000, "ms",
                            "Run of exact digital zeros inside non-silent audio (possible dropout or edit point)."))
    return out


def _spectral_jumps(m, sr):
    if len(m) < 2048:
        return []
    hop = 256
    fr = _frames(m, 1024, hop) * np.hanning(1024)
    lm = np.log(np.abs(np.fft.rfft(fr, axis=1)) + 1e-6)
    flux = np.linalg.norm(np.diff(lm, axis=0), axis=1)
    med = np.median(flux)
    scale = max(1.4826 * np.median(np.abs(flux - med)), 0.05 * med, 1e-3)
    z = (flux - med) / scale
    hits = np.flatnonzero((z > 6) & (flux > 1.0))  # flux floor ignores numerical noise on steady signals
    return [_obs("SPECTRAL_DISCONTINUITY", (i + 1) * hop / sr, (i + 1) * hop / sr, z[i], "robust_z",
                 "Abrupt frame-to-frame spectral change.") for i in hits]


def _window_checks(m, sr):
    n = sr
    wins = [m[i:i + n] for i in range(0, len(m) - n // 2, n)]
    out = []
    floors, rolls, means, levels = [], [], [], []
    for w in wins:
        floors.append(float(_db(np.percentile(frame_rms(w, sr), 10))))
        levels.append(float(_db(np.sqrt((w ** 2).mean()))))
        means.append(float(w.mean()))
        p = np.abs(np.fft.rfft(w)) ** 2
        f = np.fft.rfftfreq(len(w), 1 / sr)
        rolls.append(float(f[np.searchsorted(np.cumsum(p) / p.sum(), 0.99)]) if p.sum() > 0 else 0.0)
    for i in range(1, len(wins)):
        t = float(i)
        if abs(floors[i] - floors[i - 1]) > 10 and max(floors[i], floors[i - 1]) > -100:
            out.append(_obs("NOISE_FLOOR_JUMP", t, t, floors[i] - floors[i - 1], "dB",
                            "Background noise floor changes abruptly between adjacent seconds."))
        a, b = rolls[i - 1], rolls[i]
        if levels[i] > -50 and levels[i - 1] > -50 and a > 0 and abs(b - a) / a > 0.4:
            out.append(_obs("BANDLIMIT_CHANGE", t, t, (b - a) / a * 100, "percent",
                            "Upper bandwidth (99% rolloff) changes abruptly between adjacent seconds."))
        if abs(means[i] - means[i - 1]) > 0.02:
            out.append(_obs("DC_STEP", t, t, means[i] - means[i - 1], "linear",
                            "DC level steps between adjacent seconds."))
    return out


def _merge(obs):
    out = []
    for o in sorted(obs, key=lambda o: (o["type"], o["start"])):
        if out and out[-1]["type"] == o["type"] and o["start"] - out[-1]["end"] <= _MERGE:
            out[-1]["end"] = max(out[-1]["end"], o["end"])
            out[-1]["magnitude"] = max(out[-1]["magnitude"], o["magnitude"], key=abs)
        else:
            out.append(dict(o))
    return sorted(out, key=lambda o: (o["start"], o["type"]))


def integrity_indicators(native, sr: int) -> dict:
    m = np.atleast_2d(np.asarray(native, dtype=np.float64)).mean(axis=0)
    obs = _zero_runs(m, sr) + _spectral_jumps(m, sr) + _window_checks(m, sr)
    return {"observations": _merge(obs), "limitations": INTEGRITY_LIMITATIONS,
            "synthetic_speech_detection": dict(_SYNTHETIC)}

"""Acoustic features (numpy/scipy). Values are JSON-safe: Python floats or None, never NaN."""
import numpy as np
from scipy.signal import find_peaks

from audio_mcp.ingest.decode import resample
from audio_mcp.types import Interval

FEATURE_GROUPS = ("level", "silence", "spectral", "pitch", "rhythm", "bands")
_N_FFT, _HOP = 2048, 512
_EDGES = [0, 250, 500, 1000, 2000, 4000, 8000]
_BLOCK_FRAMES = 4096  # bounds memory: never hold the whole-file STFT
_FLOOR_AMP = 10 ** (-60 / 20)  # frames below -60 dBFS never count as active signal
_SILENCE_REL_DB = 30  # silence = more than 30 dB below the active level (never above -50 dBFS)
_SILENCE_AMP = 10 ** (-50 / 20)
_CLIP_RUN = 4  # flat-topped run length at the file's own peak that counts as clipping


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _db(num, den, k=10):
    return _f(k * np.log10(num / den)) if num > 0 and den > 0 else None


def _frames(x, size, hop):
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    n = 1 + (len(x) - size) // hop
    return np.lib.stride_tricks.as_strided(x, (n, size), (x.strides[0] * hop, x.strides[0]))


def frame_rms(x, sr, win=0.025, hop=0.010):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0)
    size = max(1, round(win * sr))
    fr = _frames(x, size, max(1, round(hop * sr)))
    out, rows = np.empty(len(fr)), max(1, (1 << 20) // size)  # float64 math per block of ~1M samples only
    for i in range(0, len(fr), rows):
        out[i:i + rows] = np.sqrt(np.square(fr[i:i + rows], dtype=np.float64).mean(axis=1))
    return out


def _clip_ratio(m, peak):
    """Share of samples at full scale, or in flat-topped runs at the file's own peak (clipped, then scaled)."""
    n, flat, block = len(m), 0, 1 << 20
    for i in range(0, n, block):
        a = np.abs(m[i:i + block + _CLIP_RUN - 1])  # overlap so runs crossing a block edge are seen
        hit = a >= 0.999
        if peak >= 0.25:
            at = a >= peak - 1e-4
            run = np.ones(len(a) - _CLIP_RUN + 1, bool) if len(a) >= _CLIP_RUN else np.zeros(0, bool)
            for k in range(_CLIP_RUN):
                run &= at[k:len(a) - _CLIP_RUN + 1 + k]
            for k in range(_CLIP_RUN):
                hit[k:k + len(run)] |= run
        flat += int(hit[:block].sum())
    return flat / n if n else 0.0


def _speech_frames(n_frames, regions, hop=0.010, win=0.025):
    centers = np.arange(n_frames) * hop + win / 2
    mask = np.zeros(n_frames, bool)
    for r in regions:
        mask |= (centers >= r.start) & (centers < r.end)
    return mask


def _level(m, native, sr, regions):
    p = np.square(m)
    rms, peak = float(np.sqrt(p.mean(dtype=np.float64))), float(np.abs(m).max())
    dc = float(m.mean(dtype=np.float64))
    v = {
        "rms": rms, "rms_dbfs": _db(rms, 1, 20), "peak_amplitude": peak,
        "per_channel_peak": [float(np.abs(c).max()) for c in native],
        "crest_factor_db": _db(peak, rms, 20),
        "clipping_ratio": _clip_ratio(m, peak), "dc_offset": dc,
    }
    fp = np.maximum(frame_rms(m, sr) ** 2 - dc ** 2, 0.0)  # frame power without the DC offset
    speech = _speech_frames(len(fp), regions) if regions else None
    active = fp[speech] if speech is not None and speech.any() else fp[fp > _FLOOR_AMP ** 2]
    v["active_rms_dbfs"] = _db(float(np.sqrt(active.mean())), 1, 20) if len(active) else None
    if speech is not None:
        pad = 5  # frames (50 ms) either side of speech still carry speech onsets/tails
        near = np.convolve(speech.astype(np.int8), np.ones(2 * pad + 1, np.int8), "same") > 0
        noise = fp[~near]
        if speech.any() and len(noise) * 0.010 >= 0.5:
            v["snr_method"] = "vad"
            v["snr_db_estimate"] = _db(float(fp[speech].mean()), float(np.median(noise)))
        else:  # continuous speech: the noise floor shows in the dips between words
            v["snr_method"] = "speech_dips"
            v["snr_db_estimate"] = _dip_snr(fp)
    else:
        v["snr_method"] = "percentile"
        v["snr_db_estimate"] = _dip_snr(fp)
    return v


def _dip_snr(fp):
    """Mean frame power over the 5th-percentile frame (digital-silence frames excluded)."""
    fp = fp[fp > 1e-12]
    return _db(float(fp.mean()), float(np.percentile(fp, 5))) if len(fp) else None


def _silence(m, sr):
    fr = frame_rms(m, sr)
    loud = fr[fr > _FLOOR_AMP]
    active = float(np.sqrt(np.mean(loud ** 2))) if len(loud) else 0.0
    quiet = fr < min(_SILENCE_AMP, max(_FLOOR_AMP, active * 10 ** (-_SILENCE_REL_DB / 20)))
    hop, win, dur = 0.010, 0.025, len(m) / sr
    ivs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            s, e = i * hop, min(dur, (j - 1) * hop + win)
            if e - s >= 0.3:
                ivs.append(Interval(start=s, end=e).model_dump())
            i = j
        else:
            i += 1
    return {"silence_ratio": float(quiet.mean()), "silence_intervals": ivs}


def _stft_blocks(m):
    """Yield STFT magnitude blocks of at most _BLOCK_FRAMES frames."""
    fr, win = _frames(m, _N_FFT, _HOP), np.hanning(_N_FFT)
    for i in range(0, len(fr), _BLOCK_FRAMES):
        yield np.abs(np.fft.rfft(fr[i:i + _BLOCK_FRAMES] * win, axis=1))


def _spectral(m, sr):
    keys = ("spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz", "spectral_flatness")
    f = np.fft.rfftfreq(_N_FFT, 1 / sr)
    tot, n = np.zeros(4), 0
    for mag in _stft_blocks(m):
        mag = mag[mag.sum(axis=1) > 1e-9]
        if not len(mag):
            continue
        w = mag / mag.sum(axis=1, keepdims=True)
        cen = (w * f).sum(axis=1)
        bw = np.sqrt((w * (f - cen[:, None]) ** 2).sum(axis=1))
        roll = f[(np.cumsum(w, axis=1) >= 0.85).argmax(axis=1)]
        flat = np.exp(np.log(mag + 1e-10).mean(axis=1)) / (mag.mean(axis=1) + 1e-10)
        tot += [cen.sum(), bw.sum(), roll.sum(), flat.sum()]
        n += len(mag)
    v = dict(zip(keys, (_f(t / n) for t in tot))) if n else dict.fromkeys(keys)
    zc = np.abs(np.diff(np.signbit(m).astype(np.int8))).mean() if len(m) > 1 else 0.0
    v["zero_crossing_rate"] = float(zc)
    return v


def _bands(m, sr):
    nyq = sr / 2
    edges = [e for e in _EDGES if e < nyq]
    f = np.fft.rfftfreq(_N_FFT, 1 / sr)
    ps, n = np.zeros(len(f)), 0
    for mag in _stft_blocks(m):
        ps += (mag ** 2).sum(axis=0)
        n += len(mag)
    ps /= n
    hi = edges[1:] + [np.inf]
    e = np.array([ps[(f >= a) & (f < b)].sum() for a, b in zip(edges, hi)])
    names = [f"{a}-{b}" for a, b in zip(edges, edges[1:] + [int(nyq)])]
    tot = e.sum()
    return {"band_energy": dict(zip(names, (e / tot if tot > 0 else e).tolist()))}


def _yin(m, sr, fmin=50, fmax=600, thr=0.15, size=1024, hop=256):
    """Return per-frame f0 (0 = unvoiced)."""
    fr = _frames(m, size, hop)
    return np.concatenate([_yin_block(fr[i:i + _BLOCK_FRAMES], sr, fmin, fmax, thr, size)
                           for i in range(0, len(fr), _BLOCK_FRAMES)])


def _yin_block(fr, sr, fmin, fmax, thr, size):
    w, tmax, tmin = size // 2, int(sr / fmin), int(sr / fmax)
    fr = fr.astype(np.float64)  # per block: keeps the YIN difference function precise
    a, nfft = fr[:, :w], 2048
    r = np.fft.irfft(np.conj(np.fft.rfft(a, nfft, axis=1)) * np.fft.rfft(fr, nfft, axis=1), nfft, axis=1)[:, :tmax + 1]
    cs = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    d = (a ** 2).sum(axis=1, keepdims=True) + (cs[:, w:w + tmax + 1] - cs[:, :tmax + 1]) - 2 * r
    d[:, 0] = 0
    cm = np.ones_like(d)
    cm[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1) / (np.cumsum(d[:, 1:], axis=1) + 1e-12)
    below = cm[:, tmin:tmax] < thr
    loud = np.sqrt((fr ** 2).mean(axis=1)) > 1e-4
    f0 = np.zeros(len(fr))
    for i in np.flatnonzero(below.any(axis=1) & loud):
        t = tmin + int(below[i].argmax())
        while t + 1 < tmax and cm[i, t + 1] < cm[i, t]:
            t += 1
        den = cm[i, t - 1] - 2 * cm[i, t] + cm[i, t + 1]
        off = 0.5 * (cm[i, t - 1] - cm[i, t + 1]) / den if abs(den) > 1e-12 else 0.0
        f0[i] = sr / (t + off)
    return f0


def _pitch(m, sr):
    y = resample(m.astype(np.float32, copy=False), sr, 16000)
    f0 = _yin(y, 16000)
    voiced = f0[f0 > 0]
    ratio = len(voiced) / len(f0)
    ok = ratio >= 0.05
    stats = {
        "f0_mean_hz": np.mean, "f0_median_hz": np.median, "f0_min_hz": np.min,
        "f0_max_hz": np.max, "f0_std_hz": np.std,
    }
    return {**{k: _f(fn(voiced)) if ok else None for k, fn in stats.items()}, "voiced_ratio": float(ratio)}


def _rhythm(m, sr):
    prev, parts = None, []
    for mag in _stft_blocks(m):
        lm = np.log1p(mag)
        if prev is not None:  # carry last frame so flux is continuous across blocks
            lm = np.vstack([prev, lm])
        prev = lm[-1:]
        parts.append(np.maximum(np.diff(lm, axis=0), 0).sum(axis=1))
    flux = np.concatenate(parts)
    fps, dur = sr / _HOP, len(m) / sr
    v = {"tempo_bpm": None, "onset_rate_per_s": 0.0}
    if len(flux) < 4 or flux.max() <= 0:
        return v
    pk, _ = find_peaks(flux, height=flux.mean() + flux.std(), distance=max(1, int(0.15 * fps)))
    v["onset_rate_per_s"] = float(len(pk) / dur)
    z = flux - flux.mean()
    ac = np.correlate(z, z, "full")[len(z) - 1:]
    lo, hi = int(60 * fps / 240), min(int(60 * fps / 40), len(ac) - 1)
    if ac[0] > 0 and hi > lo:
        lag = lo + int(ac[lo:hi + 1].argmax())
        if ac[lag] / ac[0] >= 0.1:
            v["tempo_bpm"] = float(60 * fps / lag)
    return v


_UNITS = {
    "rms": "linear", "rms_dbfs": "dBFS", "peak_amplitude": "linear", "per_channel_peak": "linear",
    "crest_factor_db": "dB", "clipping_ratio": "ratio", "dc_offset": "linear", "snr_db_estimate": "dB",
    "active_rms_dbfs": "dBFS",
    "silence_ratio": "ratio", "silence_intervals": "seconds",
    "spectral_centroid_hz": "Hz", "spectral_bandwidth_hz": "Hz", "spectral_rolloff_hz": "Hz",
    "spectral_flatness": "ratio", "zero_crossing_rate": "ratio", "band_energy": "ratio",
    "f0_mean_hz": "Hz", "f0_median_hz": "Hz", "f0_min_hz": "Hz", "f0_max_hz": "Hz", "f0_std_hz": "Hz",
    "voiced_ratio": "ratio", "tempo_bpm": "BPM", "onset_rate_per_s": "per_second",
}


def extract_features(native, sr, groups=FEATURE_GROUPS, speech_regions=None):
    # float32 throughout (half the RAM of float64); reductions accumulate in float64
    native = np.atleast_2d(np.asarray(native, dtype=np.float32))
    m = native[0] if len(native) == 1 else native.mean(axis=0, dtype=np.float32)
    fns = {
        "level": lambda: _level(m, native, sr, speech_regions), "silence": lambda: _silence(m, sr),
        "spectral": lambda: _spectral(m, sr), "pitch": lambda: _pitch(m, sr),
        "rhythm": lambda: _rhythm(m, sr), "bands": lambda: _bands(m, sr),
    }
    values = {}
    for g in groups:
        values.update(fns[g]())
    return {"units": {k: _UNITS[k] for k in values if k in _UNITS}, "values": values}

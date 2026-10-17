"""Preprocessing ops that write a derived variant; the original audio is never touched."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, istft, sosfiltfilt, stft

from audio_mcp.dsp.features import frame_rms
from audio_mcp.ingest.decode import resample

# canonical execution order: requested ops always run in this order
PREPROCESS_OPS = ("remove_dc", "mono", "highpass_80hz", "denoise", "trim_silence", "resample_16k", "normalize")
_SILENCE_AMP = 10 ** (-50 / 20)


def _remove_dc(x, sr):
    return x - x.mean(axis=1, keepdims=True), sr, {}


def _mono(x, sr):
    return x.mean(axis=0, keepdims=True), sr, {}


def _highpass(x, sr):
    sos = butter(4, 80, "highpass", fs=sr, output="sos")
    if x.shape[1] <= 3 * (2 * len(sos) + 1):  # too short for filtfilt padding
        return x, sr, {"cutoff_hz": 80, "applied": False}
    return sosfiltfilt(sos, x, axis=1), sr, {"cutoff_hz": 80, "order": 4}


def _denoise(x, sr):
    out = np.empty_like(x)
    for c, ch in enumerate(x):
        _, _, z = stft(ch, fs=sr, nperseg=1024)
        mag = np.abs(z)
        floor = np.percentile(mag, 10, axis=1, keepdims=True)
        mask = (mag > floor * 10 ** (6 / 20)).astype(np.float64)
        mask = uniform_filter1d(mask, size=3, axis=1)  # 3-frame smoothing
        _, y = istft(z * mask, fs=sr, nperseg=1024)
        out[c] = np.pad(y, (0, max(0, ch.size - y.size)))[: ch.size]
    return out, sr, {"gate_db": 6, "noise_percentile": 10}


def _trim_silence(x, sr):
    loud = np.flatnonzero(frame_rms(x, sr) >= _SILENCE_AMP)
    if not loud.size:
        return x, sr, {"offset_seconds": 0.0, "applied": False}
    hop = round(0.010 * sr)
    start, end = loud[0] * hop, min(x.shape[1], loud[-1] * hop + round(0.025 * sr))
    return x[:, start:end], sr, {"offset_seconds": round(start / sr, 3), "end_seconds": round(end / sr, 3)}


def _resample_16k(x, sr):
    if sr == 16000:
        return x, sr, {"from_hz": sr, "to_hz": 16000}
    return np.stack([resample(ch.astype(np.float32), sr, 16000) for ch in x]), 16000, {"from_hz": sr, "to_hz": 16000}


def _normalize(x, sr):
    rms, peak = float(np.sqrt((x ** 2).mean())), float(np.abs(x).max())
    if rms == 0:
        return x, sr, {"gain_db": 0.0, "applied": False}
    gain = 10 ** (-20 / 20) / rms
    limit = 10 ** (-1 / 20)
    if peak * gain > limit:
        gain = limit / peak
    return x * gain, sr, {"target_rms_dbfs": -20, "peak_limit_dbfs": -1, "gain_db": round(20 * np.log10(gain), 3)}


_OPS = {"remove_dc": _remove_dc, "mono": _mono, "highpass_80hz": _highpass, "denoise": _denoise,
        "trim_silence": _trim_silence, "resample_16k": _resample_16k, "normalize": _normalize}


def apply_preprocessing(native, sr: int, ops) -> tuple[np.ndarray, int, list[dict]]:
    unknown = set(ops) - set(PREPROCESS_OPS)
    if unknown:
        raise ValueError(f"unknown preprocessing ops: {sorted(unknown)}")
    x = np.atleast_2d(np.asarray(native, dtype=np.float64))
    transforms = []
    for op in PREPROCESS_OPS:
        if op in ops:
            x, sr, params = _OPS[op](x, sr)
            transforms.append({"op": op, "params": params})
    return x.astype(np.float32), sr, transforms

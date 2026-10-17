"""Recording-quality diagnosis: issues with measured values, thresholds and recommended preprocessing."""
import numpy as np
from scipy.signal import welch

from audio_mcp.dsp.features import _HOP, _N_FFT, _stft_blocks


def _hum_excess_db(m, sr):
    if len(m) < sr // 4:
        return None
    with np.errstate(all="ignore"):  # NaN/inf samples give a None result, not warnings
        return _hum_db(m, sr)


def _hum_db(m, sr):
    f, p = welch(m, fs=sr, nperseg=min(sr, len(m)))
    band = p[(f >= 30) & (f <= 120)]
    if not band.size or np.median(band) <= 0:
        return None
    peak = max(p[np.argmin(np.abs(f - hz))] for hz in (50, 60))
    db = 10 * np.log10(peak / np.median(band)) if peak > 0 else np.nan
    return float(db) if np.isfinite(db) else None  # NaN/inf input (e.g. float files) must not leak out


def _echo_peak(m, sr):
    """Heuristic: secondary autocorrelation peak of the onset envelope at 30-300 ms lags."""
    blocks = [np.log1p(b) for b in _stft_blocks(m)]
    lm = np.vstack(blocks)
    if len(lm) < 9:
        return None
    env = np.maximum(np.diff(lm, axis=0), 0).sum(axis=1)
    z = env - env.mean()
    if z.std() < 1e-6:
        return None
    ac = np.correlate(z, z, "full")[len(z) - 1:]
    fps = sr / _HOP
    lo, hi = max(1, int(0.03 * fps)), min(len(ac) - 1, int(0.3 * fps))
    return float(ac[lo:hi + 1].max() / ac[0]) if hi >= lo and ac[0] > 0 else None


def _rolloff99(m, sr):
    ps = sum((b ** 2).sum(axis=0) for b in _stft_blocks(m))
    if not np.any(ps):
        return None
    f = np.fft.rfftfreq(_N_FFT, 1 / sr)
    return float(f[np.searchsorted(np.cumsum(ps) / ps.sum(), 0.99)])


def _issue(code, measured, threshold, *, high, message):
    return {"code": code, "severity": "high" if high else "medium", "message": message,
            "measured": round(float(measured), 6), "threshold": threshold}


def diagnose_quality(native, sr: int, feature_values: dict, speech_regions) -> dict:
    native = np.atleast_2d(np.asarray(native, dtype=np.float64))
    m = native.mean(axis=0)
    v = feature_values
    metrics = {
        "clipping_ratio": v.get("clipping_ratio"), "rms_dbfs": v.get("rms_dbfs"),
        "silence_ratio": v.get("silence_ratio"), "snr_db_estimate": v.get("snr_db_estimate"),
        "dc_offset": v.get("dc_offset"), "hum_excess_db": _hum_excess_db(m, sr),
        "echo_autocorr_peak": _echo_peak(m, sr), "rolloff99_hz": _rolloff99(m, sr), "channels": native.shape[0],
    }
    issues, recs = [], []

    def add(issue, rec=None):
        issues.append(issue)
        if rec and rec not in recs:
            recs.append(rec)

    c = metrics["clipping_ratio"]
    if c is not None and c > 0.001:
        add(_issue("CLIPPING", c, 0.001, high=c >= 0.002,
                   message="Clipped samples detected; clipping is not recoverable by preprocessing."))
    r = metrics["rms_dbfs"]
    if r is not None and r < -35:
        add(_issue("LOW_LEVEL", r, -35, high=r <= -45, message="Overall level is low."), "normalize")
    s = metrics["silence_ratio"]
    if s is not None and s > 0.5:
        add(_issue("SILENCE", s, 0.5, high=s >= 1.0, message="Most of the recording is silent."), "trim_silence")
    snr, flat = metrics["snr_db_estimate"], v.get("spectral_flatness")
    # a steady tone also has a flat energy profile; only call it noise when it is noise-like or VAD-measured
    if snr is not None and snr < 15 and (v.get("snr_method") == "vad" or (flat is not None and flat >= 0.3)):
        add(_issue("HIGH_NOISE", snr, 15, high=snr <= 5,
                   message=f"Estimated SNR is low ({v.get('snr_method')} estimate)."), "denoise")
    h = metrics["hum_excess_db"]
    if h is not None and h >= 20:
        add(_issue("HUM", h, 20, high=h >= 30, message="Mains hum at 50/60 Hz detected."), "highpass_80hz")
    e = metrics["echo_autocorr_peak"]
    if e is not None and e > 0.5:
        add(_issue("ECHO", e, 0.5, high=False, message="Possible echo/reverberation (heuristic indicator)."))
    ro = metrics["rolloff99_hz"]
    if ro is not None and sr >= 16000 and ro < 3800:
        add(_issue("BANDWIDTH_LIMITED", ro, 3800, high=ro <= 1900,
                   message="Energy is confined below ~3.8 kHz (e.g. telephone-band audio or a narrowband source)."))
    dc = metrics["dc_offset"]
    if dc is not None and abs(dc) > 0.01:
        add(_issue("DC_OFFSET", abs(dc), 0.01, high=abs(dc) >= 0.02, message="DC offset present."), "remove_dc")
    if native.shape[0] > 1:
        i = _issue("MULTICHANNEL", native.shape[0], 1, high=False, message="Multichannel audio; models use a mono mix.")
        i["severity"] = "low"
        add(i, "mono")

    sev = {i["severity"] for i in issues}
    overall = "poor" if "high" in sev else "fair" if "medium" in sev else "good"
    return {"metrics": metrics, "issues": issues, "recommendations": recs, "overall": overall}

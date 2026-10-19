"""Recording-quality diagnosis: issues with measured values, thresholds and recommended preprocessing."""
import numpy as np
from scipy.signal import welch

from audio_mcp.dsp.features import _N_FFT, _stft_blocks

# Thresholds tuned on the development set (scripts/prepare_eval_data.py quality_dev), not the evaluation set.
_ECHO_Z = 20.0  # cepstral peak z-score: clean speech <= 8, 120 ms echo >= 80
_HF_DB = -27.0  # >4 kHz vs 300 Hz-4 kHz energy: clean speech >= -24 dB, 3.4 kHz low-pass <= -29 dB
_LOW_DB = -42.0  # active-signal level (dBFS)
_SILENCE = 0.6  # share of silent frames


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


def _echo_z(m, sr):
    """Heuristic: a discrete echo at delay d puts a peak at quefrency d in the real cepstrum. Returns the robust
    z-score of the largest 30-300 ms cepstral peak over the first 60 s (None if too short or silent)."""
    x = np.asarray(m[: 60 * sr], dtype=np.float64)
    if len(x) < sr:
        return None
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    if not np.isfinite(spec).all() or not spec.any():
        return None
    c = np.fft.irfft(np.log(spec + 1e-12 * spec.max()), len(x))
    seg = c[int(0.03 * sr):int(0.3 * sr)]
    med = np.median(seg)
    mad = 1.4826 * np.median(np.abs(seg - med))
    return float((seg.max() - med) / mad) if mad > 0 else None


def _hf_ratio_db(m, sr):
    """Energy above 4 kHz relative to 300 Hz-4 kHz, in dB (None below 16 kHz sampling or without signal)."""
    if sr < 16000:
        return None
    ps = sum((b ** 2).sum(axis=0) for b in _stft_blocks(m))
    f = np.fft.rfftfreq(_N_FFT, 1 / sr)
    lo, hi = ps[(f >= 300) & (f < 4000)].sum(), ps[f >= 4000].sum()
    if not lo > 0:
        return None
    return float(10 * np.log10(max(hi, 1e-30) / lo))


def _issue(code, measured, threshold, *, high, message):
    return {"code": code, "severity": "high" if high else "medium", "message": message,
            "measured": round(float(measured), 6), "threshold": threshold}


def diagnose_quality(native, sr: int, feature_values: dict, speech_regions) -> dict:
    native = np.atleast_2d(np.asarray(native, dtype=np.float64))
    m = native.mean(axis=0)
    v = feature_values
    metrics = {
        "clipping_ratio": v.get("clipping_ratio"), "rms_dbfs": v.get("rms_dbfs"),
        "active_rms_dbfs": v.get("active_rms_dbfs"),
        "silence_ratio": v.get("silence_ratio"), "snr_db_estimate": v.get("snr_db_estimate"),
        "dc_offset": v.get("dc_offset"), "hum_excess_db": _hum_excess_db(m, sr),
        "echo_cepstral_z": _echo_z(m, sr), "hf_energy_ratio_db": _hf_ratio_db(m, sr), "channels": native.shape[0],
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
    r = metrics["active_rms_dbfs"] if metrics["active_rms_dbfs"] is not None else metrics["rms_dbfs"]
    if r is not None and r < _LOW_DB:  # level of the active (non-silent) signal, so long pauses do not count
        add(_issue("LOW_LEVEL", r, _LOW_DB, high=r <= _LOW_DB - 10, message="Level of the active signal is low."),
            "normalize")
    s = metrics["silence_ratio"]
    if s is not None and s > _SILENCE:
        add(_issue("SILENCE", s, _SILENCE, high=s >= 1.0, message="Most of the recording is silent."), "trim_silence")
    h = metrics["hum_excess_db"]
    hum = h is not None and h >= 20
    if hum:
        add(_issue("HUM", h, 20, high=h >= 30, message="Mains hum at 50/60 Hz detected."), "highpass_80hz")
    snr, flat = metrics["snr_db_estimate"], v.get("spectral_flatness")
    # a steady tone also has a flat energy profile; only call it noise when it is noise-like or speech-measured.
    # When hum is flagged the low SNR is the hum itself: HUM (with its high-pass fix) already reports it.
    speech_based = v.get("snr_method") in ("vad", "speech_dips")
    if snr is not None and snr < 15 and not hum and (speech_based or (flat is not None and flat >= 0.3)):
        add(_issue("HIGH_NOISE", snr, 15, high=snr <= 5,
                   message=f"Estimated SNR is low ({v.get('snr_method')} estimate)."), "denoise")
    e = metrics["echo_cepstral_z"]
    if e is not None and e > _ECHO_Z:
        add(_issue("ECHO", e, _ECHO_Z, high=False,
                   message="Possible discrete echo (cepstral peak; heuristic indicator)."))
    hf = metrics["hf_energy_ratio_db"]
    if hf is not None and hf < _HF_DB:
        add(_issue("BANDWIDTH_LIMITED", hf, _HF_DB, high=hf <= _HF_DB - 20,
                   message="Almost no energy above 4 kHz (e.g. telephone-band audio or a narrowband source)."))
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

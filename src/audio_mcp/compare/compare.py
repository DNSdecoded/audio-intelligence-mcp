"""Measured differences between recordings and baseline anomaly localization (no causal claims)."""
import numpy as np

from audio_mcp.eval.metrics import _words, word_error_rate

_EDGES = [0, 250, 500, 1000, 2000, 4000, 8000]
_BAND_NAMES = [f"band_{a}-{b}" for a, b in zip(_EDGES, _EDGES[1:] + ["nyq"])]
_SCALAR_SKIP = {"per_channel_peak", "silence_intervals", "band_energy", "snr_method"}


def window_features(native, sr: int, window_seconds: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """Per non-overlapping window: level, spectral shape, ZCR and 7 band-energy fractions."""
    m = np.atleast_2d(np.asarray(native, dtype=np.float64)).mean(axis=0)
    n = max(1, int(round(window_seconds * sr)))
    names = ["rms_dbfs", "spectral_centroid_hz", "spectral_flatness", "zero_crossing_rate", *_BAND_NAMES]
    f = np.fft.rfftfreq(n, 1 / sr)
    hi = [*_EDGES[1:], sr / 2 + 1]
    rows = []
    for i in range(max(1, int(np.ceil(len(m) / n)))):
        w = m[i * n:(i + 1) * n]
        if i and len(w) < n // 4:  # drop a tiny trailing fragment
            break
        w = np.pad(w, (0, n - len(w)))
        rms = np.sqrt((w ** 2).mean())
        p = np.abs(np.fft.rfft(w * np.hanning(n))) ** 2
        tot = p.sum()
        cen = float((p * f).sum() / tot) if tot > 0 else 0.0
        flat = float(np.exp(np.log(p + 1e-20).mean()) / (p.mean() + 1e-20)) if tot > 0 else 0.0
        zcr = float(np.abs(np.diff(np.signbit(w).astype(np.int8))).mean())
        bands = [p[(f >= a) & (f < b)].sum() / tot if tot > 0 else 0.0 for a, b in zip(_EDGES, hi)]
        rows.append([max(-100.0, 20 * np.log10(rms)) if rms > 0 else -100.0, cen, flat, zcr, *bands])
    return np.array(rows, dtype=np.float64), names


def _rate(events, label, minutes):
    return round(sum(e["label"] == label for e in events) / minutes, 3) if minutes > 0 else None


def compare_pair(a: dict, b: dict) -> dict:
    va, vb = a["features"].get("values", {}), b["features"].get("values", {})
    units = {**a["features"].get("units", {}), **b["features"].get("units", {})}
    feats = {}
    for k in sorted(set(va) & set(vb) - _SCALAR_SKIP):
        x, y = va[k], vb[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            feats[k] = {"a": x, "b": y, "delta": round(y - x, 6), "unit": units.get(k)}
    ba, bb = va.get("band_energy") or {}, vb.get("band_energy") or {}
    bands = {k: round(bb.get(k, 0.0) - ba.get(k, 0.0), 6) for k in sorted(set(ba) | set(bb))}
    la, lb = {e["label"] for e in a["events"]}, {e["label"] for e in b["events"]}
    ma, mb = a["duration"] / 60, b["duration"] / 60
    events = {"only_in_a": sorted(la - lb), "only_in_b": sorted(lb - la),
              "rate_change": {lab: {"a_per_min": _rate(a["events"], lab, ma), "b_per_min": _rate(b["events"], lab, mb)}
                              for lab in sorted(la | lb)}}
    diff = {"features": feats, "bands": bands, "events": events}
    if a.get("transcript") is not None and b.get("transcript") is not None:
        wa, wb = set(_words(a["transcript"])), set(_words(b["transcript"]))
        diff["transcript"] = {"word_difference_rate": round(word_error_rate(a["transcript"], b["transcript"]), 3),
                              "a_only_words": sorted(wa - wb)[:20], "b_only_words": sorted(wb - wa)[:20]}
    sim = {}
    if a.get("embedding") is not None and b.get("embedding") is not None:
        ea, eb = np.asarray(a["embedding"]), np.asarray(b["embedding"])
        den = np.linalg.norm(ea) * np.linalg.norm(eb)
        sim["audio_embedding_cosine"] = round(float(ea @ eb / den), 4) if den > 0 else None
    return {"a": a["audio_id"], "b": b["audio_id"], "measured_differences": diff, "similarity": sim,
            "possible_interpretations": _interpret(feats, events)}


def _interpret(feats, events) -> list[dict]:
    out = []
    d = feats.get("rms_dbfs", {}).get("delta")
    if d is not None and abs(d) > 6:
        out.append({"statement": f"{'B' if d > 0 else 'A'} is louder by {abs(d):.1f} dB; this may reflect gain, "
                                 "distance, or a change in the source.", "based_on": ["rms_dbfs"]})
    c = feats.get("spectral_centroid_hz")
    if c and c["a"] and abs(c["delta"]) / c["a"] > 0.25:
        way = "higher" if c["delta"] > 0 else "lower"
        out.append({"statement": f"Spectral balance shifted toward {way} frequencies; this may reflect a different "
                                 "source, microphone, or filtering.", "based_on": ["spectral_centroid_hz"]})
    if events["only_in_b"]:
        out.append({"statement": f"Sound types present only in B: {', '.join(events['only_in_b'])}; their cause is "
                                 "not determined.", "based_on": ["events"]})
    return [{**i, "kind": "interpretation", "source": "rule"} for i in out]


def baseline_anomalies(target, baselines, *, z_threshold: float = 3.0, window_seconds: float = 1.0) -> dict:
    tf, names = target
    base = np.vstack([b[0] for b in baselines])
    mean, std = base.mean(axis=0), base.std(axis=0)
    std = np.maximum(std, 1e-6 * np.abs(mean) + 1e-9)
    file_z = (tf.mean(axis=0) - mean) / std
    wz = (tf - mean) / std
    hot = np.abs(wz) >= z_threshold
    windows, i = [], 0
    while i < len(tf):
        if not hot[i].any():
            i += 1
            continue
        j = i
        while j + 1 < len(tf) and hot[j + 1].any():
            j += 1
        block = wz[i:j + 1]
        peak = block[np.abs(block).argmax(axis=0), np.arange(len(names))]
        windows.append({"start": round(i * window_seconds, 3), "end": round((j + 1) * window_seconds, 3),
                        "features": {n: round(float(z), 2) for n, z in zip(names, peak) if abs(z) >= z_threshold}})
        i = j + 1
    order = np.argsort(-np.abs(file_z))
    return {"feature_z": {names[k]: round(float(file_z[k]), 2) for k in range(len(names))},
            "anomalous_features": [names[k] for k in order if abs(file_z[k]) >= z_threshold],
            "anomalous_windows": windows, "baseline_windows": int(len(base)), "z_threshold": z_threshold,
            "method": "per-window z-score against the pooled baseline window distribution"}

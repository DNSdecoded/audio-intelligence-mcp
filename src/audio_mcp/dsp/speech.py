"""Speaking rate, pauses and pitch variability from word timings and features. Emotion/tone is not inferred."""
_PAUSE, _LONG_PAUSE = 0.25, 1.0


def _wpm(words: int, seconds: float):
    return round(words / seconds * 60, 1) if words and seconds > 0 else None


def speech_characteristics(results: dict) -> dict:
    segs = (results.get("speaker_assign") or results.get("transcription") or {}).get("segments", [])
    words = sorted((w for s in segs for w in s.get("words", [])), key=lambda w: w["start"])
    speech = sum(r["end"] - r["start"] for r in (results.get("vad") or {}).get("speech_regions", []))
    pauses = [(a["end"], b["start"]) for a, b in zip(words, words[1:]) if b["start"] - a["end"] >= _PAUSE]
    pause_s = sum(e - s for s, e in pauses)
    v = (results.get("features") or {}).get("values") or {}
    lo, hi = v.get("f0_min_hz"), v.get("f0_max_hz")
    out = {
        "words_per_minute": _wpm(len(words), speech),
        "articulation_rate_wpm": _wpm(len(words), speech - pause_s),
        "pause_count": len(pauses),
        "mean_pause_seconds": round(pause_s / len(pauses), 3) if pauses else None,
        "long_pauses": [{"start": round(s, 3), "end": round(e, 3)} for s, e in pauses if e - s >= _LONG_PAUSE],
        "f0_range_hz": round(hi - lo, 1) if lo is not None and hi is not None else None,
        "f0_std_hz": v.get("f0_std_hz"),
        "notes": "emotion and tone are not inferred",
    }
    by: dict[str, list] = {}
    for w in words:
        if w.get("speaker_id"):
            by.setdefault(w["speaker_id"], []).append(w)
    if by:
        out["per_speaker"] = {
            spk: {"words": len(ws), "words_per_minute": _wpm(len(ws), sum(w["end"] - w["start"] for w in ws))}
            for spk, ws in by.items()}
    return out

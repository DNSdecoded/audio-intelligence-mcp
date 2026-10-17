"""Uncertainty and conflict reporting over transcript, VAD, tagger and quality results."""


def _overlap(a0, a1, regions):
    return sum(max(0.0, min(a1, r["end"]) - max(a0, r["start"])) for r in regions)


def _item(type_, start, end, detail, ids=()):
    return {"type": type_, "start": round(float(start), 3), "end": round(float(end), 3), "detail": detail,
            "evidence_ids": list(ids)}


def _low_word_runs(seg, low):
    run = []
    for w in [*seg.get("words", []), None]:
        if w is not None and w.get("confidence") is not None and w["confidence"] < low:
            run.append(w)
            continue
        if len(run) >= 2:
            yield _item("LOW_CONFIDENCE_WORDS", run[0]["start"], run[-1]["end"],
                        f"{len(run)} consecutive words below {low}", [seg["id"]])
        run = []


def find_uncertainties(results: dict, *, low_conf: float, second_pass: list[dict] | None = None) -> dict:
    segs = (results.get("transcription") or {}).get("segments", [])
    vad = (results.get("vad") or {}).get("speech_regions", [])
    items, n_words, n_low = [], 0, 0
    for s in segs:
        c = s.get("confidence")
        if c is not None and c < low_conf:
            items.append(_item("LOW_CONFIDENCE_SEGMENT", s["start"], s["end"], f"segment confidence {c:.2f}", [s["id"]]))
        items += _low_word_runs(s, low_conf)
        conf = [w["confidence"] for w in s.get("words", []) if w.get("confidence") is not None]
        n_words, n_low = n_words + len(conf), n_low + sum(x < low_conf for x in conf)
        dur = s["end"] - s["start"]
        if dur > 0 and _overlap(s["start"], s["end"], vad) < 0.2 * dur:
            items.append(_item("ASR_WITHOUT_VAD", s["start"], s["end"],
                               "transcribed text where VAD found little speech", [s["id"]]))
    for r in vad:
        if r["end"] - r["start"] >= 1.0 and _overlap(r["start"], r["end"], segs) == 0:
            items.append(_item("VAD_WITHOUT_ASR", r["start"], r["end"], "speech detected but nothing transcribed"))
    if "events" in results:
        speech_ev = [e for e in results["events"]["events"] if e["label"] == "Speech"]
        for e in speech_ev:
            d = e["end"] - e["start"]
            if d >= 1.0 and _overlap(e["start"], e["end"], vad) < 0.2 * d:
                items.append(_item("TAGGER_VAD_DISAGREEMENT", e["start"], e["end"],
                                   "sound tagger heard speech that VAD did not", [e["id"]]))
        for r in vad:
            if r["end"] - r["start"] >= 1.0 and _overlap(r["start"], r["end"], speech_ev) == 0:
                items.append(_item("TAGGER_VAD_DISAGREEMENT", r["start"], r["end"],
                                   "VAD found speech that the sound tagger did not"))
    for q in (results.get("quality") or {}).get("issues", []):
        if q["severity"] == "high":
            items.append(_item("QUALITY_AFFECTED", q.get("start", 0.0), q.get("end", 0.0),
                               f"{q['code']}: {q['message']}"))
    cross = None
    if second_pass is not None:
        from audio_mcp.eval.metrics import word_error_rate

        primary = " ".join(s["text"] for s in segs)
        cross = {"word_disagreement_rate": round(word_error_rate(primary, " ".join(s["text"] for s in second_pass)), 3),
                 "method": "beam-5 vs greedy t=0.4 decode"}
    items.sort(key=lambda i: (i["start"], i["type"]))
    conflicts = ("ASR_WITHOUT_VAD", "VAD_WITHOUT_ASR", "TAGGER_VAD_DISAGREEMENT")
    return {"items": items, "summary": {
        "low_confidence_word_ratio": round(n_low / n_words, 3) if n_words else None,
        "conflict_count": sum(i["type"] in conflicts for i in items),
        "asr_cross_check": cross}}

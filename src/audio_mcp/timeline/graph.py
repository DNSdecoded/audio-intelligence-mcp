"""Temporal event graph: nodes for speech, turns, events, silence, overlaps, transitions; relation edges."""
import bisect

NODE_TYPES = ("speech_segment", "speaker_turn", "sound_event", "silence", "overlap", "transition")
_ONSET_MIN_CONF = 0.5
_LEVEL_JUMP_DB = 12.0


def _node(id_, type_, label, start, end, source, *, speaker_id=None, confidence=None, kind=None):
    n = {"id": id_, "type": type_, "label": label, "start": round(float(start), 3), "end": round(float(end), 3),
         "source": source}
    if speaker_id:
        n["speaker_id"] = speaker_id
    if confidence is not None:
        n.update(confidence=confidence, confidence_kind=kind)
    return n


def _nodes(results, level_db):
    nodes, trn = [], []
    segs = (results.get("speaker_assign") or results.get("transcription") or {}).get("segments", [])
    for s in segs:
        nodes.append(_node(s["id"], "speech_segment", s["text"], s["start"], s["end"], "transcription",
                           speaker_id=s.get("speaker_id"), confidence=s.get("confidence"), kind=s.get("confidence_kind")))
    diar = results.get("diarization") or {}
    turns = diar.get("turns") or []
    for i, t in enumerate(turns, 1):
        nodes.append(_node(f"turn_{i:04d}", "speaker_turn", t["speaker_id"], t["start"], t["end"], "diarization",
                           speaker_id=t["speaker_id"]))
    for a, b in zip(turns, turns[1:]):
        if a["speaker_id"] != b["speaker_id"]:
            trn.append((b["start"], f"speaker_change: {a['speaker_id']}->{b['speaker_id']}", "diarization"))
    for i, o in enumerate(diar.get("overlaps") or [], 1):
        nodes.append(_node(f"ovl_{i:04d}", "overlap", "overlapping speech", o["start"], o["end"], "diarization"))
    for e in (results.get("events") or {}).get("events", []):
        if segs and e["label"] == "Speech":  # the transcript already covers speech
            continue
        nodes.append(_node(e["id"], "sound_event", e["label"], e["start"], e["end"], e.get("source", "events"),
                           confidence=e.get("confidence"), kind=e.get("confidence_kind")))
        if (e.get("confidence") or 0) >= _ONSET_MIN_CONF and e["label"] not in ("Speech", "Silence"):
            trn.append((e["start"], f"onset: {e['label']}", e.get("source", "events")))
    sil = ((results.get("features") or {}).get("values") or {}).get("silence_intervals") or []
    for i, s in enumerate(sil, 1):
        nodes.append(_node(f"sil_{i:04d}", "silence", "silence", s["start"], s["end"], "features"))
    for i in range(1, len(level_db or [])):
        a, b = level_db[i - 1], level_db[i]
        if a is not None and b is not None and abs(b - a) > _LEVEL_JUMP_DB:
            trn.append((float(i), f"level_jump: {b - a:+.1f} dB", "features"))
    for i, (t, label, src) in enumerate(sorted(trn), 1):
        nodes.append(_node(f"trn_{i:04d}", "transition", label, t, t, src))
    nodes.sort(key=lambda n: (n["start"], n["end"], n["id"]))
    return nodes


def _edges(nodes):
    spans = [n for n in nodes if n["type"] != "transition"]
    starts = [n["start"] for n in spans]
    edges = []
    for n in spans:  # nearest following node
        j = bisect.bisect_left(starts, n["end"])
        while j < len(spans) and spans[j] is n:
            j += 1
        if j < len(spans):
            edges.append({"source": n["id"], "target": spans[j]["id"], "relation": "precedes",
                          "gap_seconds": round(spans[j]["start"] - n["end"], 3)})
    active = []
    for b in spans:  # sweep: only pairs that actually intersect
        active = [a for a in active if a["end"] > b["start"]]
        for a in active:
            if b["end"] <= a["end"]:
                inner, outer = b, a
            elif a["start"] == b["start"] and a["end"] <= b["end"]:
                inner, outer = a, b
            else:
                edges.append({"source": a["id"], "target": b["id"], "relation": "overlaps"})
                continue
            rel = "interrupts" if inner["type"] == "sound_event" and outer["type"] == "speech_segment" else "during"
            edges.append({"source": inner["id"], "target": outer["id"], "relation": rel})
        active.append(b)
    return edges


def _ts(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def render_text(tl: dict) -> str:
    lines = []
    for n in tl["nodes"]:
        line = f"{_ts(n['start'])}–{_ts(n['end'])}  {n['type']}  {n['label']}"
        if n.get("speaker_id") and n["type"] != "speaker_turn":
            line += f" ({n['speaker_id']})"
        if n.get("confidence") is not None:
            line += f" ({n['confidence']:.2f})"
        lines.append(line)
    return "\n".join(lines)


def build_timeline(results: dict, duration: float, *, include: set[str] | None = None,
                   level_db: list[float | None] | None = None) -> dict:
    nodes = [n for n in _nodes(results, level_db) if include is None or n["type"] in include]
    tl = {"duration_seconds": round(float(duration), 3), "nodes": nodes, "edges": _edges(nodes)}
    tl["text"] = render_text(tl)
    return tl

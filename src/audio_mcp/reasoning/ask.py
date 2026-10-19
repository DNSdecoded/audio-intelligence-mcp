"""Rule-based, evidence-grounded question answering over cached analysis results."""
import re

import numpy as np

from audio_mcp.analysis.events import EVENT_CATEGORIES

_STOP = set("""a an the is are was were be been of to in on at by for with and or it this that there any some
what when where who how does did do speaker speakers someone something mention mentions said say says talk about
discuss recording audio sound sounds hear heard""".split())
_MIN_SCORE = 0.35
_WINDOW = 5.0  # seconds before/after an event that count as "immediately"

_INTENTS = [
    ("unusual", re.compile(r"unusual|strange|\bodd\b|anomal|unexpected")),
    ("speakers", re.compile(r"how many (speakers|people|voices)|who (speaks|talks)")),
    ("time_of_mention", re.compile(
        r"when (?:does|did|is|do) .*?(?:mention|says|say|talk about|discuss)(?:es)? (?P<topic>.+?)\??$")),
    ("time_of_mention", re.compile(  # passive: "when is the deadline mentioned (in the call)?"
        r"(?:when|at what time) (?:is|was|are|were) (?P<topic>.+?) "
        r"(?:mentioned|discussed|said|talked about|brought up|raised)\b.*$")),
    ("relative_to_event", re.compile(
        r"(?P<rel>before|after|during|while) (?:the )?(?P<target>[a-z ,]+?)"
        r"(?: sound| sounds| starts| begins| goes off| rings)?\??$")),
    ("presence", re.compile(r"^(?:is|are) there (?:any |a |an )?(?P<target>.+?)\??$")),
]


def classify_intent(q: str) -> tuple[str, dict]:
    q = q.strip().lower()
    for name, rx in _INTENTS:
        m = rx.search(q)
        if m:
            return name, {k: v.strip() for k, v in m.groupdict().items() if v}
    if re.search(r"chang|differ", q) and re.search(r"beginning|start", q) and "end" in q:
        return "change", {}
    return "fallback", {}


def _tokens(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        out.add(re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 4 else w)
    return out


def resolve_labels(target: str, detected: list[str]) -> list[str]:
    t = _tokens(target)
    found = []
    for key, labels in EVENT_CATEGORIES.items():
        for label in labels if _tokens(key) & t else []:
            found.append(label)
    for label in {lab for labels in EVENT_CATEGORIES.values() for lab in labels} | set(detected):
        if _tokens(label) & t:
            found.append(label)
    return list(dict.fromkeys(found))


def _ts(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _ev(node: dict, kind: str = "detected") -> dict:
    e = {"id": node["id"], "start": node["start"], "end": node["end"], "source": node.get("source", "analysis"),
         "kind": kind}
    if node.get("type") == "speech_segment" or "text" in node:
        e["text"] = node.get("text", node.get("label"))
    else:
        e["label"] = node.get("label")
    if node.get("confidence") is not None:
        e.update(confidence=node["confidence"], confidence_kind=node.get("confidence_kind", "model_probability"))
    return e


def _answer(text, support, intent, evidence):
    out = {"answer": text, "support": support, "intent": intent, "evidence": evidence}
    conf = [e for e in evidence if e["kind"] == "detected" and e.get("confidence") is not None]
    if support != "insufficient" and conf:
        best = max(conf, key=lambda e: e["confidence"])
        out.update(confidence=best["confidence"], confidence_kind=best["confidence_kind"])
    return out


def _segments(results):
    return (results.get("speaker_assign") or results.get("transcription") or {}).get("segments", [])


def _events(results):
    return (results.get("events") or {}).get("events", [])


def _nodes(results):
    return [n for n in (results.get("timeline") or {}).get("nodes", []) if n["type"] != "transition"]


def _scored(query, items, text_of, embedder):
    q = _tokens(query)
    texts = [text_of(i) for i in items]
    kw = [len(q & _tokens(t)) / len(q) if q else 0.0 for t in texts]
    if embedder is not None and items:
        v = embedder.embed([query] + texts)
        cos = [float(x) for x in v[1:] @ v[0]]
        score = [0.5 * k + 0.5 * max(c, 0.0) for k, c in zip(kw, cos)]
    else:
        score = kw
    return sorted(zip(score, kw, items), key=lambda x: -x[0])


def _mention(p, results, embedder):
    ranked = [r for r in _scored(p["topic"], _segments(results), lambda s: s["text"], embedder) if r[0] >= _MIN_SCORE]
    if not ranked:
        return _answer(f"The recording does not contain a clear mention of {p['topic']}.", "insufficient",
                       "time_of_mention", [])
    keep = sorted(ranked[:5], key=lambda r: r[2]["start"])
    ev = [_ev({**s, "source": "transcription"}) for _, _, s in keep]
    when = ", ".join(f"{_ts(s['start'])}–{_ts(s['end'])}" for _, _, s in keep)
    support = "direct" if any(k > 0 for _, k, _ in keep) else "inferred"
    return _answer(f"{p['topic'].capitalize()} is mentioned at {when}.", support, "time_of_mention", ev)


def _relative(p, results):
    events = _events(results)
    if "events" not in results:
        return _answer("Sound event detection was not available, so this cannot be answered.", "insufficient",
                       "relative_to_event", [])
    labels = set(resolve_labels(p["target"], [e["label"] for e in events]))
    matched = sorted((e for e in events if e["label"] in labels), key=lambda e: e["start"])
    if not matched:
        return _answer(f"No {p['target']} was detected in the recording.", "insufficient", "relative_to_event", [])
    e, rel = matched[0], p["rel"]
    nodes = [n for n in _nodes(results) if n["id"] != e["id"]]
    if rel == "before":
        near = [n for n in nodes if n["start"] < e["start"] and n["end"] >= e["start"] - _WINDOW]
    elif rel == "after":
        near = [n for n in nodes if n["end"] > e["end"] and n["start"] <= e["end"] + _WINDOW]
    else:
        linked = {(x["source"], x["target"]) for x in (results.get("timeline") or {}).get("edges", [])
                  if x["relation"] in ("during", "interrupts", "overlaps")}
        near = [n for n in nodes if (n["id"], e["id"]) in linked or (e["id"], n["id"]) in linked]
    anchor = {**e, "type": "sound_event"}
    if not near:
        return _answer(f"Nothing else was detected {rel} {e['label']} at {_ts(e['start'])}.", "direct",
                       "relative_to_event", [_ev(anchor)])
    parts = "; ".join(f"{n['type'].replace('_', ' ')} '{n['label']}' {_ts(n['start'])}–{_ts(n['end'])}" for n in near)
    return _answer(f"{rel.capitalize()} {e['label']} at {_ts(e['start'])}: {parts}.", "direct", "relative_to_event",
                   [_ev(anchor)] + [_ev(n) for n in near])


def _presence(p, results):
    if "events" not in results:
        return _answer("Sound event detection was not available.", "insufficient", "presence", [])
    events = _events(results)
    labels = set(resolve_labels(p["target"], [e["label"] for e in events]))
    found = [e for e in events if e["label"] in labels]
    thr = results["events"].get("threshold")
    if not found:
        end = (results.get("metadata") or {}).get("duration_seconds", 0.0)
        return _answer(f"No {p['target']} detected (threshold {thr}).", "direct", "presence",
                       [{"id": "detector", "start": 0.0, "end": end, "source": "events", "label": "<none>",
                         "kind": "detected"}])
    when = ", ".join(f"{e['label']} {_ts(e['start'])}–{_ts(e['end'])}" for e in found[:10])
    return _answer(f"Yes: {when}.", "direct", "presence", [_ev({**e, "type": "sound_event"}) for e in found])


def _unusual(results, duration):
    events = _events(results)
    total: dict[str, float] = {}
    for e in events:
        total[e["label"]] = total.get(e["label"], 0.0) + e["end"] - e["start"]
    common = set(sorted(total, key=lambda k: -total[k])[:3])
    rare = [e for e in events if e["label"] not in common and total[e["label"]] < 0.1 * max(duration, 1e-9)]
    ev = [_ev({**e, "type": "sound_event"}, "interpretation") for e in rare]
    for q in (results.get("quality") or {}).get("issues", []):
        if q["severity"] in ("medium", "high"):
            ev.append({"id": f"quality:{q['code']}", "start": 0.0, "end": duration, "source": "quality",
                       "label": q["message"], "kind": "interpretation"})
    for o in (results.get("integrity") or {}).get("observations", []):
        ev.append({"id": f"integrity:{o['type']}", "start": o["start"], "end": o["end"], "source": "integrity",
                   "label": o["description"], "kind": "interpretation"})
    if not ev:
        return _answer("Nothing unusual was detected.", "inferred", "unusual", [])
    return _answer("Possibly unusual: " + "; ".join(e.get("label") or e.get("text", "") for e in ev[:8]) + ".",
                   "inferred", "unusual", ev)


def _third_stats(audio, lo, hi):
    x = audio.mono16k[int(lo * 16000):int(hi * 16000)]
    if not x.size:
        return None, None
    rms = float(np.sqrt((x ** 2).mean()))
    seg = x[: 16000 * 30]
    mag = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(seg.size, 1 / 16000)
    cen = float((mag * f).sum() / mag.sum()) if mag.sum() > 0 else None
    return (round(20 * np.log10(rms), 1) if rms > 0 else None), (round(cen) if cen else None)


def _change(results, duration, audio):
    a, b = duration / 3, 2 * duration / 3

    def labels(lo, hi, items, key):
        return sorted({i[key] for i in items if i["start"] < hi and i["end"] > lo})

    parts = []
    if audio is not None:
        (l1, c1), (l2, c2) = _third_stats(audio, 0, a), _third_stats(audio, b, duration)
        if l1 is not None and l2 is not None:
            parts.append(f"level {l1} → {l2} dBFS ({l2 - l1:+.1f} dB)")
        if c1 and c2:
            parts.append(f"spectral centroid {c1} → {c2} Hz")
    e1, e2 = labels(0, a, _events(results), "label"), labels(b, duration, _events(results), "label")
    if set(e1) != set(e2):
        parts.append(f"sounds {e1} → {e2}")
    turns = (results.get("diarization") or {}).get("turns", [])
    s1, s2 = labels(0, a, turns, "speaker_id"), labels(b, duration, turns, "speaker_id")
    if turns and s1 != s2:
        parts.append(f"speakers {s1} → {s2}")
    ev = [{"id": "first_third", "start": 0.0, "end": round(a, 3), "source": "features", "label": "first third",
           "kind": "detected"},
          {"id": "last_third", "start": round(b, 3), "end": round(duration, 3), "source": "features",
           "label": "last third", "kind": "detected"}]
    if not parts:
        return _answer("No measurable change between the beginning and the end.", "direct", "change", ev)
    return _answer("Measured change from the first to the last third: " + "; ".join(parts) + ".", "direct",
                   "change", ev)


def _speakers(results):
    d = results.get("diarization")
    if not d:
        return _answer("Speaker diarization was not available for this recording.", "insufficient", "speakers", [])
    sp = d["speakers"]
    desc = ", ".join(f"{s['speaker_id']} ({s['total_speech_seconds']:.1f} s)" for s in sp)
    ev = [{"id": f"turn_{i:04d}", "start": t["start"], "end": t["end"], "source": "diarization",
           "label": t["speaker_id"], "kind": "detected"} for i, t in enumerate(d["turns"], 1)]
    return _answer(f"{len(sp)} speaker{'s' if len(sp) != 1 else ''} detected: {desc}.", "direct", "speakers", ev)


def _fallback(question, results, embedder):
    items = [{**s, "source": "transcription", "_t": s["text"]} for s in _segments(results)]
    items += [{**e, "type": "sound_event", "_t": e["label"]} for e in _events(results)]
    ranked = _scored(question, items, lambda i: i["_t"], embedder)
    if not ranked or ranked[0][0] < _MIN_SCORE:
        return _answer("The recording does not contain enough evidence to answer this question.", "insufficient",
                       "fallback", [])
    top = [i for s, _, i in ranked[:3] if s >= _MIN_SCORE]
    ev = [_ev({k: v for k, v in i.items() if k != "_t"}, "interpretation") for i in top]
    return _answer("Most relevant parts: " + "; ".join(f"{_ts(i['start'])} {i['_t']}" for i in top) + ".",
                   "inferred", "fallback", ev)


def answer_question(question: str, results: dict, duration: float, text=None, *, audio=None) -> dict:
    intent, p = classify_intent(question)
    if intent == "time_of_mention":
        return _mention(p, results, text)
    if intent == "relative_to_event":
        return _relative(p, results)
    if intent == "presence":
        return _presence(p, results)
    if intent == "unusual":
        return _unusual(results, duration)
    if intent == "change":
        return _change(results, duration, audio)
    if intent == "speakers":
        return _speakers(results)
    return _fallback(question, results, text)

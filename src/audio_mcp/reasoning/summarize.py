"""Extractive, evidence-linked summaries: every sentence is a transcript segment or a detected event."""
import math

import numpy as np

from audio_mcp.reasoning.ask import _tokens

SENTENCES = {"short": 2, "medium": 4, "long": 8}
_SKIP_EVENTS = {"Speech", "Silence", "Noise"}  # generic labels add nothing to a summary


def _ts(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _tfidf(docs: list[list[str]]) -> list[dict]:
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(len(docs) / c) + 1 for t, c in df.items()}
    return [{t: d.count(t) * idf[t] for t in set(d)} for d in docs]


def _key_moments(results, segs) -> list[dict]:
    km = []
    if segs:
        km.append({"timestamp": segs[0]["start"], "description": "Speech begins", "evidence_ids": [segs[0]["id"]]})
    turns = (results.get("diarization") or {}).get("turns", [])
    for i, (a, b) in enumerate(zip(turns, turns[1:]), 2):
        if a["speaker_id"] != b["speaker_id"]:
            km.append({"timestamp": b["start"], "description": f"{b['speaker_id']} starts speaking",
                       "evidence_ids": [f"turn_{i:04d}"]})
    for e in (results.get("events") or {}).get("events", []):
        if e["label"] not in _SKIP_EVENTS and (e.get("confidence") or 0) >= 0.5:
            km.append({"timestamp": e["start"], "description": f"{e['label']} detected", "evidence_ids": [e["id"]]})
    return sorted(km, key=lambda k: k["timestamp"])[:10]


def _nonspeech(results, duration) -> dict:
    events = [e for e in (results.get("events") or {}).get("events", []) if e["label"] not in _SKIP_EVENTS]
    sentences = [{"text": f"{duration:.1f} s recording with no transcribed speech.", "evidence_ids": []}]
    for e in sorted(events, key=lambda e: -(e.get("confidence") or 0))[:5]:
        sentences.append({"text": f"Detected {e['label']} ({_ts(e['start'])}–{_ts(e['end'])}).",
                          "evidence_ids": [e["id"]]})
    sil = ((results.get("features") or {}).get("values") or {}).get("silence_ratio")
    if sil:
        sentences.append({"text": f"Silence covers {sil:.0%} of the recording.", "evidence_ids": ["features"]})
    grounded = [s for s in sentences if s["evidence_ids"]] or sentences
    return {"text": " ".join(s["text"] for s in sentences), "sentences": grounded, "topics": [],
            "key_moments": _key_moments(results, []), "method": "extractive"}


def summarize(results: dict, duration: float, text=None, *, length: str = "short", focus: str | None = None) -> dict:
    segs = (results.get("speaker_assign") or results.get("transcription") or {}).get("segments", [])
    segs = [s for s in segs if s["text"].strip()]
    if not segs:
        return _nonspeech(results, duration)
    texts = [s["text"] for s in segs]
    weights = _tfidf([sorted(_tokens(t)) for t in texts])
    sal = np.array([sum(w.values()) / max(len(w), 1) for w in weights])
    sal = sal / sal.max() if sal.max() > 0 else sal
    cent = np.zeros(len(segs))
    emb = text.embed(texts + ([focus] if focus else [])) if text is not None else None
    if emb is not None and len(segs) > 1:
        sim = emb[: len(segs)] @ emb[: len(segs)].T
        cent = (sim.sum(axis=1) - np.diag(sim)) / (len(segs) - 1)
    score = 0.6 * cent + 0.4 * sal
    if focus:
        if emb is not None:
            score = score + 0.5 * np.clip(emb[: len(segs)] @ emb[-1], 0, None)
        else:
            f = _tokens(focus)
            score = score + 0.5 * np.array([len(f & _tokens(t)) / max(len(f), 1) for t in texts])
    n = min(SENTENCES[length], len(segs))
    pick = sorted(np.argsort(-score, kind="stable")[:n])
    sentences = [{"text": segs[i]["text"].strip(), "evidence_ids": [segs[i]["id"]]} for i in pick]
    salient = [e for e in (results.get("events") or {}).get("events", [])
               if e["label"] not in _SKIP_EVENTS and (e.get("confidence") or 0) >= 0.5]
    for e in sorted(salient, key=lambda e: -e["confidence"])[:3]:  # notable sounds alongside the speech
        sentences.append({"text": f"Detected {e['label']} ({_ts(e['start'])}–{_ts(e['end'])}).",
                          "evidence_ids": [e["id"]]})
    totals: dict[str, float] = {}
    for w in weights:
        for t, v in w.items():
            totals[t] = totals.get(t, 0.0) + v
    topics = [{"label": t, "evidence_ids": [s["id"] for s, w in zip(segs, weights) if t in w]}
              for t in sorted(totals, key=lambda t: (-totals[t], t))[:5]]
    return {"text": " ".join(s["text"] for s in sentences), "sentences": sentences, "topics": topics,
            "key_moments": _key_moments(results, segs), "method": "extractive"}

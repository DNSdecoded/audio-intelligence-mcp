"""LLM answers, summaries and interpretations constrained to an evidence pack; uncited claims are dropped."""
import json
import re

from audio_mcp.reasoning.ask import _scored, _segments
from audio_mcp.reasoning.summarize import _key_moments
from audio_mcp.types import Evidence

_PACK_SEGMENTS, _PACK_MAX = 20, 40
_SYSTEM = ("You analyse audio recordings using only the evidence provided. Every claim must cite evidence ids. "
           "If the evidence is insufficient, say so. Reply with a single JSON object and nothing else.")


def build_evidence_pack(results: dict, query: str, text=None) -> list[Evidence]:
    segs = [r[2] for r in _scored(query, _segments(results), lambda s: s["text"], text)[:_PACK_SEGMENTS]]
    events = [e for e in (results.get("events") or {}).get("events", []) if (e.get("confidence") or 0) >= 0.5]
    pack = [Evidence(id=s["id"], start=s["start"], end=s["end"], source="transcription", text=s["text"],
                     confidence=s.get("confidence"), confidence_kind=s.get("confidence_kind")) for s in segs]
    pack += [Evidence(id=e["id"], start=e["start"], end=e["end"], source=e.get("source", "events"), label=e["label"],
                      confidence=e.get("confidence"), confidence_kind=e.get("confidence_kind")) for e in events]
    return sorted(pack, key=lambda e: e.start)[:_PACK_MAX]


def _lines(pack) -> str:
    rows = []
    for e in pack:
        d = {"id": e.id, "start": e.start, "end": e.end, "source": e.source}
        d["text" if e.text is not None else "label"] = e.text if e.text is not None else e.label
        rows.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(rows)


def _parse(out: str) -> dict:
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in LLM output")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    return obj


def _list(v) -> list:
    """LLM JSON is untrusted: a field that should be a list may be anything."""
    return v if isinstance(v, list) else []


def _cites(item, ids) -> bool:
    """A dict whose non-empty evidence_ids are all strings naming evidence in the pack."""
    v = item.get("evidence_ids") if isinstance(item, dict) else None
    return isinstance(v, list) and bool(v) and all(isinstance(i, str) and i in ids for i in v)


def llm_answer(question: str, pack: list[Evidence], llm) -> dict:
    prompt = (f"Evidence (JSON lines):\n{_lines(pack)}\n\nQuestion: {question}\n\nReply as JSON: "
              '{"answer": str, "claims": [{"text": str, "evidence_ids": [str], "kind": "detected"|"interpretation"}], '
              '"insufficient": bool}')
    obj = _parse(llm.complete(_SYSTEM, prompt))
    by_id = {e.id: e for e in pack}
    claims = [c for c in _list(obj.get("claims")) if _cites(c, by_id)]
    cited = list(dict.fromkeys(i for c in claims for i in c["evidence_ids"]))
    if obj.get("insufficient") or not claims:
        support = "insufficient"
    else:
        support = "direct" if any(c.get("kind") == "detected" for c in claims) else "inferred"
    evidence = [{**by_id[i].model_dump(exclude_none=True),
                 "kind": "detected" if any(c.get("kind") == "detected" and i in c["evidence_ids"] for c in claims)
                 else "interpretation"} for i in cited]
    answer = obj.get("answer") or "The recording does not contain enough evidence to answer this question."
    out = {"answer": str(answer), "support": support, "intent": "llm", "evidence": evidence,
           "claims": claims, "method": "llm"}
    conf = [e for e in evidence if e["kind"] == "detected" and e.get("confidence") is not None]
    if support != "insufficient" and conf:
        best = max(conf, key=lambda e: e["confidence"])
        out.update(confidence=best["confidence"], confidence_kind=best["confidence_kind"])
    return out


def llm_summarize(results: dict, pack: list[Evidence], llm, length: str, focus: str | None) -> dict:
    prompt = (f"Evidence (JSON lines):\n{_lines(pack)}\n\nWrite a {length} summary"
              + (f" focused on: {focus}" if focus else "") + ". Reply as JSON: "
              '{"sentences": [{"text": str, "evidence_ids": [str]}], "topics": [str]}')
    obj = _parse(llm.complete(_SYSTEM, prompt))
    ids = {e.id for e in pack}
    sentences = [{"text": s["text"], "evidence_ids": s["evidence_ids"]} for s in _list(obj.get("sentences"))
                 if _cites(s, ids) and isinstance(s.get("text"), str) and s["text"].strip()]
    topics = [{"label": t, "evidence_ids": []} for t in _list(obj.get("topics")) if isinstance(t, str)][:5]
    return {"text": " ".join(s["text"] for s in sentences), "sentences": sentences, "topics": topics,
            "key_moments": _key_moments(results, _segments(results)), "method": "llm"}


def llm_interpret(measured_differences: dict, llm) -> list[dict]:
    prompt = (f"Measured differences between recording A and B:\n{json.dumps(measured_differences)}\n\n"
              "Suggest at most 3 possible, clearly hedged interpretations. Reply as JSON: "
              '{"interpretations": [{"statement": str, "based_on": [str]}]}')
    obj = _parse(llm.complete(_SYSTEM, prompt))
    return [{"statement": i["statement"], "based_on": [b for b in _list(i.get("based_on")) if isinstance(b, str)],
             "kind": "interpretation", "source": "llm"} for i in _list(obj.get("interpretations"))[:3]
            if isinstance(i, dict) and isinstance(i.get("statement"), str) and i["statement"].strip()]

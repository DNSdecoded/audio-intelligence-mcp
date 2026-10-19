"""Evaluation metrics: diarization error rate (frame-based) and word error rate."""
import unicodedata
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_rttm(path) -> list[dict]:
    turns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            turns.append({"speaker_id": f[7], "start": start, "end": round(start + dur, 3)})
    return turns


def _matrix(turns, labels, n, frame):
    m = np.zeros((n, len(labels)), dtype=bool)
    for t in turns:
        m[int(round(t["start"] / frame)):int(round(t["end"] / frame)), labels.index(t["speaker_id"])] = True
    return m


def der(ref: list[dict], hyp: list[dict], *, collar: float = 0.25, frame: float = 0.01) -> dict:
    """Frame-based DER with an optimal one-to-one speaker mapping; ±collar around reference boundaries is unscored."""
    end = max([t["end"] for t in ref + hyp], default=0.0)
    n = int(np.ceil(end / frame)) + 1
    rl, hl = sorted({t["speaker_id"] for t in ref}), sorted({t["speaker_id"] for t in hyp})
    R, H = _matrix(ref, rl, n, frame), _matrix(hyp, hl, n, frame)
    keep = np.ones(n, dtype=bool)
    c = int(round(collar / frame))
    if c:
        for t in ref:
            for b in (t["start"], t["end"]):
                i = int(round(b / frame))
                keep[max(0, i - c):i + c] = False
    R, H = R[keep], H[keep]
    nr, nh = R.sum(axis=1), H.sum(axis=1)
    total = int(nr.sum())
    if total == 0:
        return {"der": 0.0, "missed": 0.0, "false_alarm": 0.0, "confusion": 0.0}
    correct = 0
    if rl and hl:
        overlap = R.T.astype(np.int64) @ H.astype(np.int64)
        ri, hi = linear_sum_assignment(-overlap)
        correct = int(overlap[ri, hi].sum())
    missed = int(np.maximum(nr - nh, 0).sum())
    fa = int(np.maximum(nh - nr, 0).sum())
    conf = int(np.minimum(nr, nh).sum()) - correct
    return {"der": (missed + fa + conf) / total, "missed": missed / total, "false_alarm": fa / total,
            "confusion": conf / total}


def _words(text: str) -> list[str]:
    # strip punctuation/symbols by Unicode category; \W would also strip combining marks (Devanagari, Arabic vowels)
    return "".join(" " if unicodedata.category(c)[0] in "PS" and c != "'" else c for c in text.lower()).split()


def word_error_rate(ref: str, hyp: str) -> float:
    """Word-level Levenshtein distance / reference length (lowercased, punctuation stripped)."""
    return _edit_rate(_words(ref), _words(hyp))


def char_error_rate(ref: str, hyp: str) -> float:
    """Character-level error rate over the same normalization, spaces removed (for unsegmented scripts)."""
    return _edit_rate(list("".join(_words(ref))), list("".join(_words(hyp))))


def _edit_rate(r: list[str], h: list[str]) -> float:
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def _iou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def event_prf(ref: list[dict], hyp: list[dict], *, iou: float = 0.5) -> dict:
    """Event-level precision/recall/F1: same label (case-insensitive; a reference label "A|B" accepts either),
    greedy one-to-one matching by IoU."""
    pairs = sorted(((_iou(r, h), i, j) for i, r in enumerate(ref) for j, h in enumerate(hyp)
                    if h["label"].lower() in r["label"].lower().split("|")), reverse=True)
    used_r, used_h = set(), set()
    for v, i, j in pairs:
        if v >= iou and i not in used_r and j not in used_h:
            used_r.add(i)
            used_h.add(j)
    tp = len(used_r)
    fp, fn = len(hyp) - tp, len(ref) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}

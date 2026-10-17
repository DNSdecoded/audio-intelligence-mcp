"""Speaker clustering over embedding windows and speaker assignment to transcript segments."""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from audio_mcp.types import Interval, Segment, SpeakerTurn

_WIN, _HOP, _SR = 1.5, 0.75, 16000
_MIN_CLUSTER = 0.8  # shorter windows embed unreliably: they join the nearest cluster instead


def _windows(regions: list[Interval]) -> list[tuple[float, float]]:
    out = []
    for r in regions:
        if r.end - r.start <= _WIN:
            out.append((r.start, r.end))
            continue
        starts = list(np.arange(r.start, r.end - _WIN + 1e-9, _HOP))
        if starts[-1] + _WIN < r.end:  # last window reaches the region end
            starts.append(r.end - _WIN)
        out += [(float(s), float(s) + _WIN) for s in starts]
    return out


def _labels(emb: np.ndarray, threshold: float, num_speakers: int | None, max_speakers: int) -> np.ndarray:
    if len(emb) < 2:
        return np.ones(len(emb), dtype=int)
    z = linkage(emb, "average", metric="cosine")
    if num_speakers:
        return fcluster(z, t=num_speakers, criterion="maxclust")
    lab = fcluster(z, t=threshold, criterion="distance")
    if len(set(lab)) > max_speakers:
        lab = fcluster(z, t=max_speakers, criterion="maxclust")
    return lab


def cluster_speakers(mono16k, regions, embedder, *, threshold, num_speakers, max_speakers) -> list[SpeakerTurn]:
    wins = _windows(regions)
    if not wins:
        return []
    emb = np.asarray(embedder.embed([mono16k[int(s * _SR):int(e * _SR)] for s, e in wins]), dtype=np.float64)
    long = np.array([e - s >= _MIN_CLUSTER for s, e in wins])
    if long.sum() >= 2 and not long.all():
        labels = np.zeros(len(wins), dtype=int)
        labels[long] = _labels(emb[long], threshold, num_speakers, max_speakers)
        unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        cents = {lab: unit[labels == lab].mean(axis=0) for lab in set(labels[long])}
        for i in np.flatnonzero(~long):
            labels[i] = max(cents, key=lambda lab: float(unit[i] @ cents[lab]))
    else:
        labels = _labels(emb, threshold, num_speakers, max_speakers)
    turns: list[list] = []  # [start, end, raw_label]
    for (s, e), lab in zip(wins, labels):
        if turns and s <= turns[-1][1]:  # overlapping/touching window
            if lab == turns[-1][2]:
                turns[-1][1] = max(turns[-1][1], e)
                continue
            mid = (s + turns[-1][1]) / 2  # boundary at the overlap midpoint
            turns[-1][1] = mid
            s = mid
        turns.append([s, e, lab])
    names: dict = {}
    for t in turns:
        names.setdefault(t[2], f"speaker_{len(names) + 1}")
    return [SpeakerTurn(speaker_id=names[lab], start=s, end=e) for s, e, lab in turns if e > s]


def _best(start, end, turns):
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(end, t.end) - max(start, t.start)
        if ov > best_ov:
            best, best_ov = t.speaker_id, ov
    return best


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    out = []
    for s in segments:
        words = [w.model_copy(update={"speaker_id": _best(w.start, w.end, turns)}) for w in s.words]
        out.append(s.model_copy(update={"speaker_id": _best(s.start, s.end, turns), "words": words}))
    return out


def speaker_summary(turns: list[SpeakerTurn]) -> list[dict]:
    by: dict[str, dict] = {}
    for t in turns:
        d = by.setdefault(t.speaker_id, {"speaker_id": t.speaker_id, "first_seen": t.start, "last_seen": t.end,
                                         "total_speech_seconds": 0.0})
        d["last_seen"] = max(d["last_seen"], t.end)
        d["total_speech_seconds"] += t.end - t.start
    return [{**d, "total_speech_seconds": round(d["total_speech_seconds"], 3)} for d in by.values()]

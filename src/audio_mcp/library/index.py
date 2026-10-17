"""Persistent audio library: SQLite FTS5 keyword search + brute-force vector search over segments."""
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm", ".aiff", ".aif", ".wma"}
_CHUNK = 50_000  # rows per vector-scoring chunk
_CANDIDATES = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings(audio_id TEXT PRIMARY KEY, path TEXT, sha256 TEXT, size_bytes INTEGER,
  quick_hash TEXT, duration REAL, language TEXT, speaker_count INTEGER, labels TEXT, indexed_at REAL, embedding BLOB);
CREATE TABLE IF NOT EXISTS segments(rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, audio_id TEXT, kind TEXT, start REAL,
  end REAL, text TEXT, label TEXT, speaker_id TEXT, embedding BLOB);
CREATE INDEX IF NOT EXISTS segments_audio ON segments(audio_id);
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(text, content='segments', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
  INSERT INTO segments_fts(rowid, text) VALUES (new.rowid, new.text); END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
  INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.rowid, old.text); END;
"""


def _quick_hash(path: Path) -> str:
    """sha256 of the first and last MiB: a cheap change check at search time."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
        size = path.stat().st_size
        if size > 1 << 20:
            f.seek(max(1 << 20, size - (1 << 20)))
            h.update(f.read())
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _blob(v) -> bytes | None:
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


class Library:
    def __init__(self, db_path: Path, text=None):
        self.db_path, self.text = Path(db_path), text
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)

    def close(self):
        self._db.close()

    def add(self, record, source_path: Path, results: dict, audio_embedding) -> int:
        aid = record.audio_id
        segs = (results.get("speaker_assign") or results.get("transcription") or {}).get("segments", [])
        events = (results.get("events") or {}).get("events", [])
        rows = [(f"{aid}:{s['id']}", aid, "speech", s["start"], s["end"], s["text"], None, s.get("speaker_id"))
                for s in segs]
        rows += [(f"{aid}:{e['id']}", aid, "event", e["start"], e["end"], e["label"], e["label"], None) for e in events]
        embs = self.text.embed([r[5] for r in rows]) if self.text is not None and rows else [None] * len(rows)
        diar = results.get("diarization") or {}
        rec = (aid, str(source_path), _sha256(source_path), source_path.stat().st_size, _quick_hash(source_path),
               record.probe.duration_seconds, (results.get("transcription") or {}).get("language"),
               len(diar["speakers"]) if diar.get("speakers") else None,
               json.dumps(sorted({e["label"] for e in events})), time.time(), _blob(audio_embedding))
        with self._lock, self._db:
            self._db.execute("DELETE FROM segments WHERE audio_id=?", (aid,))
            self._db.execute("INSERT OR REPLACE INTO recordings VALUES (?,?,?,?,?,?,?,?,?,?,?)", rec)
            self._db.executemany("INSERT INTO segments(id,audio_id,kind,start,end,text,label,speaker_id,embedding) "
                                 "VALUES (?,?,?,?,?,?,?,?,?)", [(*r, _blob(e)) for r, e in zip(rows, embs)])
        return len(rows)

    def remove(self, audio_id: str) -> bool:
        with self._lock, self._db:
            self._db.execute("DELETE FROM segments WHERE audio_id=?", (audio_id,))
            return self._db.execute("DELETE FROM recordings WHERE audio_id=?", (audio_id,)).rowcount > 0

    def remove_older_than(self, cutoff: float) -> int:
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT audio_id FROM recordings WHERE indexed_at < ?", (cutoff,))]
        return sum(self.remove(a) for a in ids)

    def list_recordings(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT audio_id, path, duration, language, speaker_count, labels, indexed_at "
                                    "FROM recordings ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
                                    (limit, offset)).fetchall()
        return [{**dict(r), "labels": json.loads(r["labels"])} for r in rows]

    def _allowed(self, filters: dict) -> dict:
        where, args = [], []
        for key, col, op in (("language", "language", "="), ("min_duration", "duration", ">="),
                             ("max_duration", "duration", "<="), ("speaker_count", "speaker_count", "=")):
            if filters.get(key) is not None:
                where.append(f"{col} {op} ?")
                args.append(filters[key])
        if filters.get("audio_ids"):
            where.append(f"audio_id IN ({','.join('?' * len(filters['audio_ids']))})")
            args += filters["audio_ids"]
        sql = "SELECT * FROM recordings" + (" WHERE " + " AND ".join(where) if where else "")
        with self._lock:
            recs = {r["audio_id"]: r for r in self._db.execute(sql, args)}
        lab = filters.get("event_label")
        if lab:
            recs = {k: r for k, r in recs.items() if lab.lower() in (x.lower() for x in json.loads(r["labels"]))}
        return recs

    def _fts(self, query: str) -> dict[int, float]:
        toks = "".join(c if c.isalnum() else " " for c in query.lower()).split()
        if not toks:
            return {}
        q = " OR ".join(f'"{t}"' for t in toks)
        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT rowid FROM segments_fts WHERE segments_fts MATCH ? ORDER BY bm25(segments_fts) LIMIT ?",
                (q, _CANDIDATES))]
        return {rid: 1 / (1 + i) for i, rid in enumerate(ids)}

    def _vector(self, query: str) -> dict[int, float]:
        if self.text is None:
            return {}
        qv = _unit(self.text.embed([query])[0])
        best: list[tuple[float, int]] = []
        last = 0
        while True:  # keyset pagination keeps memory bounded for large libraries
            with self._lock:
                rows = self._db.execute("SELECT rowid, embedding FROM segments WHERE embedding IS NOT NULL "
                                        "AND rowid > ? ORDER BY rowid LIMIT ?", (last, _CHUNK)).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            m = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
            if m.shape[1] == qv.shape[0]:
                sims = _unit(m) @ qv
                best = sorted(best + [(float(s), r[0]) for s, r in zip(sims, rows)], reverse=True)[:_CANDIDATES]
        return {rid: max(s, 0.0) for s, rid in best if s > 0}

    def search(self, query: str | None, *, filters: dict | None = None, limit: int = 10,
               query_audio_embedding=None) -> list[dict]:
        recs = self._allowed(filters or {})
        if not recs:
            return []
        audio = {}
        if query_audio_embedding is not None:
            qa = _unit(np.asarray(query_audio_embedding, dtype=np.float32))
            for aid, r in recs.items():
                if r["embedding"] is not None:
                    v = np.frombuffer(r["embedding"], dtype=np.float32)
                    if v.shape == qa.shape:
                        audio[aid] = round(float(_unit(v) @ qa), 4)
        scores: dict[int, dict] = {}
        if query:
            for name, part in (("bm25", self._fts(query)), ("text_cosine", self._vector(query))):
                for rid, s in part.items():
                    scores.setdefault(rid, {})[name] = round(s, 4)
        checked: dict[str, bool] = {}
        out = []
        if scores:
            with self._lock:
                rows = self._db.execute(
                    f"SELECT rowid, * FROM segments WHERE rowid IN ({','.join('?' * len(scores))})",
                    list(scores)).fetchall()
            for r in rows:
                if r["audio_id"] not in recs:
                    continue
                sc = dict(scores[r["rowid"]])
                if r["audio_id"] in audio:
                    sc["audio_cosine"] = audio[r["audio_id"]]
                sc["combined"] = round(sum(sc.values()) / len(sc), 4)
                out.append(self._match(recs[r["audio_id"]], r, sc, checked))
        elif audio:  # query by example only: rank whole recordings
            for aid, s in audio.items():
                out.append(self._match(recs[aid], None, {"audio_cosine": s, "combined": s}, checked))
        return sorted(out, key=lambda m: -m["scores"]["combined"])[:limit]

    def _match(self, rec, seg, scores, checked) -> dict:
        aid = rec["audio_id"]
        if aid not in checked:
            p = Path(rec["path"])
            checked[aid] = p.is_file() and p.stat().st_size == rec["size_bytes"] and _quick_hash(p) == rec["quick_hash"]
        m = {"audio_id": aid, "path": rec["path"], "scores": scores, "source_available": checked[aid]}
        if seg is None:
            return {**m, "segment_id": None, "kind": "recording", "start": 0.0, "end": rec["duration"], "excerpt": None}
        m.update(segment_id=seg["id"], kind=seg["kind"], start=seg["start"], end=seg["end"], excerpt=seg["text"])
        if seg["label"]:
            m["label"] = seg["label"]
        if seg["speaker_id"]:
            m["speaker_id"] = seg["speaker_id"]
        return m

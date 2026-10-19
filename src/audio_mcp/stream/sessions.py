"""Chunked streaming sessions: provisional per-chunk results, then one full analysis and a reconciliation."""
import base64
import binascii
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_mcp.analysis.events import frames_to_events
from audio_mcp.dsp.features import extract_features
from audio_mcp.envelope import AudioMCPError, ErrorCode
from audio_mcp.eval.metrics import word_error_rate
from audio_mcp.ingest.decode import decode, probe, resample
from audio_mcp.ingest.sources import check_path_allowed
from audio_mcp.pipeline.runner import model_slot

_ASR_WINDOW = 30.0
_IOU = 0.3


@dataclass
class Session:
    id: str
    dir: Path
    chunk_format: str
    sample_format: str | None
    sr: int | None
    channels: int | None
    samples: int = 0
    chunks: list = field(default_factory=list)
    words_until: float = 0.0
    provisional_text: list = field(default_factory=list)
    provisional_events: list = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # serializes push/finish/close
    closed: bool = False

    @property
    def buffer(self) -> Path:
        return self.dir / "native.f32"

    @property
    def duration(self) -> float:
        return self.samples / self.sr if self.sr else 0.0


def _iou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


class StreamSessions:
    def __init__(self, app):
        self.app = app
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _get(self, session_id) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise AudioMCPError(ErrorCode.INVALID_INPUT, "unknown session")
        s.last_activity = time.time()
        return s

    def start(self, chunk_format, sample_rate_hz=None, channels=None, sample_format=None) -> dict:
        if chunk_format == "pcm" and not (sample_rate_hz and channels and sample_format):
            raise AudioMCPError(ErrorCode.INVALID_INPUT, "pcm sessions need sample_rate_hz, channels and sample_format")
        sid = f"str_{uuid.uuid4().hex[:12]}"
        d = self.app.settings.process_temp_dir / f"stream_{sid}"
        d.mkdir(parents=True)
        with self._lock:
            self._sessions[sid] = Session(sid, d, chunk_format, sample_format, sample_rate_hz, channels)
        return {"session_id": sid, "state": "open"}

    def _decode_chunk(self, s: Session, data_b64, path) -> np.ndarray:
        st = self.app.settings
        if (data_b64 is None) == (path is None):
            raise AudioMCPError(ErrorCode.INVALID_INPUT, "push needs exactly one of data or path")
        raw = b""
        if data_b64 is not None:
            if len(data_b64) * 3 / 4 > st.max_upload_mb * 2 ** 20:
                raise AudioMCPError(ErrorCode.FILE_TOO_LARGE, "chunk exceeds max_upload_mb")
            try:
                raw = base64.b64decode(data_b64, validate=True)
            except (binascii.Error, ValueError):
                raise AudioMCPError(ErrorCode.INVALID_INPUT, "chunk is not valid base64") from None
        if s.chunk_format == "pcm":
            raw = raw if data_b64 is not None else check_path_allowed(path, st.allowed_roots).read_bytes()
            dtype, scale = (np.int16, 32768.0) if s.sample_format == "s16le" else (np.float32, 1.0)
            if len(raw) % (np.dtype(dtype).itemsize * s.channels):
                raise AudioMCPError(ErrorCode.INVALID_INPUT, "chunk is not a whole number of PCM frames")
            return (np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale).reshape(-1, s.channels).T
        f = check_path_allowed(path, st.allowed_roots) if path is not None else s.dir / f"chunk_{len(s.chunks)}.bin"
        if data_b64 is not None:
            f.write_bytes(raw)
        info = probe(f, st)
        x = decode(f, info, st).native
        if s.sr is None:
            s.sr, s.channels = info.sample_rate_hz, x.shape[0]
        if info.sample_rate_hz != s.sr:
            x = np.stack([resample(ch, info.sample_rate_hz, s.sr) for ch in x])
        if x.shape[0] != s.channels:
            x = np.repeat(x.mean(axis=0, keepdims=True), s.channels, axis=0)
        return x.astype(np.float32)

    def push(self, session_id, data_b64=None, path=None) -> dict:
        s = self._get(session_id)
        with s.lock:
            if s.closed:
                raise AudioMCPError(ErrorCode.INVALID_INPUT, "unknown session")
            return self._push(s, data_b64, path)

    def _push(self, s: Session, data_b64, path) -> dict:
        x = self._decode_chunk(s, data_b64, path)
        if (s.samples + x.shape[1]) / s.sr > self.app.settings.max_duration_seconds:
            raise AudioMCPError(ErrorCode.DURATION_TOO_LONG, "stream exceeds max_duration_seconds")
        t0 = s.duration
        with s.buffer.open("ab") as fh:
            fh.write(np.ascontiguousarray(x.T).tobytes())
        s.samples += x.shape[1]
        t1 = s.duration
        s.chunks.append((t0, t1))
        out = {"chunk_index": len(s.chunks) - 1, "chunk_start": round(t0, 3), "chunk_end": round(t1, 3),
               "provisional": True, "features": extract_features(x, s.sr, ["level"])["values"], "events": []}
        b = self.app.backends
        mono = resample(x.mean(axis=0), s.sr, 16000) if s.sr != 16000 else x.mean(axis=0)
        with model_slot(self.app):  # per-chunk tagging and ASR count against max_concurrent_jobs
            if b.tagger.available()[0]:
                ev = frames_to_events(b.tagger.tag_windows(mono), duration=t1 - t0,
                                      threshold=self.app.settings.event_threshold)
                out["events"] = [{**e.model_dump(exclude_none=True), "start": round(e.start + t0, 3),
                                  "end": round(e.end + t0, 3)} for e in ev]
                s.provisional_events += out["events"]
            if b.asr.available()[0]:
                out["transcript_window"] = self._asr_window(s)
        return out

    def _asr_window(self, s: Session) -> dict:
        start = max(0.0, s.duration - _ASR_WINDOW)
        with s.buffer.open("rb") as fh:
            fh.seek(int(start * s.sr) * s.channels * 4)
            tail = np.frombuffer(fh.read(), dtype=np.float32).reshape(-1, s.channels).mean(axis=1)
        mono = resample(tail, s.sr, 16000) if s.sr != 16000 else tail
        segs = []
        for seg in self.app.backends.asr.transcribe(mono).segments:
            d = seg.model_dump(exclude_none=True)
            d["start"], d["end"] = round(d["start"] + start, 3), round(d["end"] + start, 3)
            segs.append(d)
            if d["start"] >= s.words_until:  # only text not already reported by an earlier window
                s.provisional_text.append(d["text"])
                s.words_until = d["end"]
        return {"start": round(start, 3), "end": round(s.duration, 3), "text": " ".join(x["text"] for x in segs),
                "segments": segs}

    def status(self, session_id) -> dict:
        s = self._get(session_id)
        return {"session_id": s.id, "chunks": len(s.chunks), "duration_seconds": round(s.duration, 3), "state": "open"}

    def finish_audio(self, session_id):
        """Write the stream to FLAC and ingest it; returns (record, session) for the caller's full analysis."""
        s = self._get(session_id)
        with s.lock:
            if s.closed:
                raise AudioMCPError(ErrorCode.INVALID_INPUT, "unknown session")
            if not s.samples:
                raise AudioMCPError(ErrorCode.EMPTY_AUDIO, "no audio was pushed to this session")
            data = np.fromfile(s.buffer, dtype=np.float32).reshape(-1, s.channels)
            out = s.dir / "stream.flac"
            sf.write(out, np.clip(data, -1.0, 1.0), s.sr, subtype="PCM_24")
            return self.app.store.ingest_path(out, f"stream-{s.id}.flac"), s

    def reconcile(self, s: Session, analysis: dict) -> dict:
        final_text = (analysis.get("transcription") or {}).get("text", "")
        final_ev = analysis.get("sound_events", [])
        confirmed, added = [], []
        for e in final_ev:
            match = any(p["label"] == e["label"] and _iou(p, e) >= _IOU for p in s.provisional_events)
            (confirmed if match else added).append({"label": e["label"], "start": e["start"], "end": e["end"]})
        dropped = [{"label": p["label"], "start": p["start"], "end": p["end"]} for p in s.provisional_events
                   if not any(p["label"] == e["label"] and _iou(p, e) >= _IOU for e in final_ev)]
        provisional = " ".join(s.provisional_text)
        wdr = word_error_rate(final_text, provisional) if final_text or provisional else 0.0
        return {"transcript_word_difference_rate": round(wdr, 3), "events_confirmed": confirmed,
                "events_added": added, "events_dropped": dropped}

    def close(self, session_id) -> dict:
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s is None:
            raise AudioMCPError(ErrorCode.INVALID_INPUT, "unknown session")
        with s.lock:  # wait for an in-flight push/finish before deleting its files
            s.closed = True
            shutil.rmtree(s.dir, ignore_errors=True)
        return {"session_id": session_id, "state": "cancelled"}

    def expire(self, idle_seconds: float = 3600) -> int:
        cutoff = time.time() - idle_seconds
        with self._lock:
            old = [sid for sid, s in self._sessions.items() if s.last_activity < cutoff]
        for sid in old:
            self.close(sid)
        return len(old)

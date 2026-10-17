import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_mcp.config import Settings
from audio_mcp.envelope import AudioMCPError, ErrorCode
from audio_mcp.ingest.decode import DecodedAudio, ProbeInfo, decode, probe
from audio_mcp.ingest.sources import AudioIdSource, AudioSource, materialize_source

_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}")
_ID_RE = re.compile(r"aud_[0-9a-f]{16}")
_LRU_SIZE = 2


@dataclass
class AudioRecord:
    audio_id: str
    path: Path
    probe: ProbeInfo
    original_name: str
    ingested_at: float
    last_used: float = 0.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _atomic_place(write, final: Path) -> None:
    """Write via a uniquely named temp file, then rename; a concurrent writer of the same content may win."""
    tmp = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        try:
            os.replace(tmp, final)
        except PermissionError:  # Windows: the target is open (another writer placed it first)
            if not final.exists():
                raise
    finally:
        tmp.unlink(missing_ok=True)


_TOUCH_INTERVAL = 60.0  # seconds; bounds meta.json rewrites


def _probe_from_dict(d: dict) -> ProbeInfo:
    d = dict(d)
    d["format_name"] = d.pop("format")
    return ProbeInfo(**d)


class AudioStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir / "audio"
        self._cache: OrderedDict[str, DecodedAudio] = OrderedDict()
        self._lock = threading.Lock()

    def _dir(self, audio_id: str) -> Path:
        if not _ID_RE.fullmatch(audio_id):  # also blocks path traversal
            raise AudioMCPError(ErrorCode.AUDIO_NOT_FOUND, "unknown audio_id")
        return self.root / audio_id

    def ingest(self, src: AudioSource) -> AudioRecord:
        if isinstance(src, AudioIdSource):
            return self.get(src.audio_id)
        m = materialize_source(src, self.settings)
        try:
            return self.ingest_path(m.path, m.original_name)
        finally:
            if m.is_temp:
                m.path.unlink(missing_ok=True)

    def ingest_path(self, path: Path, original_name: str) -> AudioRecord:
        info = probe(path, self.settings)  # reject bad files before storing anything
        audio_id = "aud_" + _sha256(path)[:16]
        d = self._dir(audio_id)
        suffix = Path(original_name).suffix
        ext = suffix if _EXT_RE.fullmatch(suffix) else ".bin"
        meta_path = d / "meta.json"
        if not meta_path.exists():
            d.mkdir(parents=True, exist_ok=True)
            _atomic_place(lambda t: shutil.copyfile(path, t), d / f"source{ext}")
            now = time.time()
            meta = {
                "audio_id": audio_id,
                "original_name": original_name,
                "ingested_at": now,
                "last_used": now,
                "probe": info.to_dict(),
                "source_ext": ext,
            }
            if not meta_path.exists():  # a concurrent ingest of the same bytes may have finished meanwhile
                _atomic_place(lambda t: t.write_text(json.dumps(meta), encoding="utf-8"), meta_path)
        return self.get(audio_id)

    def get(self, audio_id: str, *, touch: bool = True) -> AudioRecord:
        """touch=True records use (retention sweeps on last_used); sweeps read with touch=False."""
        d = self._dir(audio_id)
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise AudioMCPError(ErrorCode.AUDIO_NOT_FOUND, "unknown audio_id") from None
        now = time.time()
        if touch and now - meta.get("last_used", meta["ingested_at"]) >= _TOUCH_INTERVAL:
            meta["last_used"] = now
            try:
                _atomic_place(lambda t: t.write_text(json.dumps(meta), encoding="utf-8"), d / "meta.json")
            except OSError:  # best effort: a lost touch only makes the record expire sooner
                pass
        return AudioRecord(
            audio_id,
            d / f"source{meta['source_ext']}",
            _probe_from_dict(meta["probe"]),
            meta["original_name"],
            meta["ingested_at"],
            meta.get("last_used", meta["ingested_at"]),
        )

    def load(self, audio_id: str) -> DecodedAudio:
        with self._lock:
            if audio_id in self._cache:
                self._cache.move_to_end(audio_id)
                return self._cache[audio_id]
        rec = self.get(audio_id)
        audio = decode(rec.path, rec.probe, self.settings)
        with self._lock:
            self._cache[audio_id] = audio
            self._cache.move_to_end(audio_id)
            while len(self._cache) > _LRU_SIZE:
                self._cache.popitem(last=False)
        return audio

    def put_variant(
        self, audio_id: str, name: str, native: np.ndarray, sr: int, transforms: list[dict]
    ) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
            raise AudioMCPError(ErrorCode.INVALID_INPUT, "invalid variant name")
        self.get(audio_id)
        vdir = self._dir(audio_id) / "variants"
        vdir.mkdir(parents=True, exist_ok=True)
        out = vdir / f"{name}.flac"
        sf.write(str(out), np.asarray(native, dtype=np.float32).T, sr, format="FLAC")
        (vdir / f"{name}.json").write_text(
            json.dumps({"sample_rate_hz": sr, "transforms": transforms}), encoding="utf-8"
        )
        return out

    def delete(self, audio_id: str) -> None:
        shutil.rmtree(self._dir(audio_id), ignore_errors=True)
        with self._lock:
            self._cache.pop(audio_id, None)

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "meta.json").exists())

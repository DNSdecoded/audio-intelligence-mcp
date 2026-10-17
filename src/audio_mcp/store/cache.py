import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from audio_mcp.config import Settings


def config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


class ResultCache:
    def __init__(self, settings: Settings):
        self.root = settings.data_dir / "results"

    def _path(self, audio_id: str, stage: str, cfg_hash: str) -> Path:
        return self.root / audio_id / f"{stage}-{cfg_hash}.json"

    def get(self, audio_id: str, stage: str, cfg_hash: str) -> dict | None:
        try:
            return json.loads(self._path(audio_id, stage, cfg_hash).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def put(self, audio_id: str, stage: str, cfg_hash: str, value: dict) -> None:
        p = self._path(audio_id, stage, cfg_hash)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        os.replace(tmp, p)

    def clear(self, audio_id: str) -> None:
        shutil.rmtree(self.root / audio_id, ignore_errors=True)

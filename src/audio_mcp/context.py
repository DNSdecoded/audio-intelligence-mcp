from dataclasses import dataclass
from typing import Any

from audio_mcp.backends.base import Backends, build_backends
from audio_mcp.config import Settings
from audio_mcp.store.audio_store import AudioStore
from audio_mcp.store.cache import ResultCache


@dataclass
class AppContext:
    settings: Settings
    store: AudioStore
    cache: ResultCache
    backends: Backends
    jobs: Any = None  # filled by later tasks
    library: Any = None
    streams: Any = None


def build_app_context(settings: Settings) -> AppContext:
    return AppContext(settings, AudioStore(settings), ResultCache(settings), build_backends(settings))

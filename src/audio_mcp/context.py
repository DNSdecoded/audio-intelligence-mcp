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
    from audio_mcp.jobs import JobManager
    from audio_mcp.library.index import Library

    from audio_mcp.stream.sessions import StreamSessions

    backends = build_backends(settings)
    text = backends.text if backends.text.available()[0] else None
    app = AppContext(settings, AudioStore(settings), ResultCache(settings), backends,
                      jobs=JobManager(settings.max_concurrent_jobs),
                     library=Library(settings.data_dir / "library.sqlite", text))
    app.streams = StreamSessions(app)
    return app

import time

import pytest

from conftest import fp
from audio_mcp.envelope import AudioMCPError
from audio_mcp.store.retention import cleanup_temp, sweep
from audio_mcp.tools.handlers import extract_audio_features, search_audio_library
from audio_mcp.tools.schemas import FeaturesRequest, SearchLibraryRequest

LATER = 25 * 3600


def _ingest(app, p):
    return extract_audio_features(app, FeaturesRequest(source=fp(p)))["audio_id"]


def test_ttl_sweep(app, speech_en):
    aid = _ingest(app, speech_en)
    assert sweep(app)["audio_deleted"] == 0
    out = sweep(app, now=time.time() + LATER)
    assert out["audio_deleted"] == 1 and out["results_deleted"] >= 1
    with pytest.raises(AudioMCPError):
        app.store.get(aid)
    assert app.cache.get(aid, "features", "x") is None


def test_keep_and_none(app, speech_en):
    _ingest(app, speech_en)
    app.settings.retention = "keep"
    assert sweep(app, now=time.time() + LATER)["audio_deleted"] == 0
    app.settings.retention = "none"
    assert sweep(app)["audio_deleted"] == 0
    assert sweep(app, final=True)["audio_deleted"] == 1


def test_library_survives_sweep(app, speech_en):
    search_audio_library(app, SearchLibraryRequest(action="index", paths=[str(speech_en)]))
    sweep(app, now=time.time() + LATER)
    assert search_audio_library(app, SearchLibraryRequest(action="list"))["result"]["recordings"]
    app.settings.library_retention = "ttl"
    assert sweep(app, now=time.time() + LATER)["library_rows_deleted"] == 1


def test_cleanup_temp(settings):
    t = settings.resolved_temp_dir
    (t / "a.bin").write_bytes(b"x")
    (t / "sub").mkdir()
    (t / "sub" / "b.bin").write_bytes(b"y")
    assert cleanup_temp(settings) == 2 and not any(t.iterdir())

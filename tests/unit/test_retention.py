import time
from pathlib import Path

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


def test_cleanup_temp_only_removes_dead_instance_dirs(settings):
    t = settings.resolved_temp_dir
    (t / "a.bin").write_bytes(b"x")  # unrelated files (temp_dir may be a shared folder) are never touched
    (t / "sub").mkdir()
    (t / "sub" / "b.bin").write_bytes(b"y")
    dead = t / "proc_999999999_deadbeef"
    dead.mkdir()
    (dead / "z.bin").write_bytes(b"z")
    assert cleanup_temp(settings) == 1 and not dead.exists()
    assert (t / "a.bin").exists() and (t / "sub" / "b.bin").exists()


@pytest.mark.parametrize("field", ["temp_dir", "export_dir", "models_dir", "ffmpeg_path", "ffprobe_path"])
def test_blank_path_settings_mean_default(field, monkeypatch):
    from audio_mcp.config import Settings

    monkeypatch.setenv("AUDIO_MCP_" + field.upper(), "")
    s = Settings()
    assert getattr(s, field) is None and s.resolved_temp_dir != Path(".")


def test_reuse_refreshes_last_used(app, speech_en, monkeypatch):  # I7
    t0 = time.time()
    aid = _ingest(app, speech_en)
    monkeypatch.setattr(time, "time", lambda: t0 + 20 * 3600)
    _ingest(app, speech_en)  # re-ingest 20 h later touches the record
    monkeypatch.undo()
    assert sweep(app, now=t0 + LATER)["audio_deleted"] == 0  # 5 h after last use, ttl 24 h
    assert app.store.get(aid).audio_id == aid
    assert sweep(app, now=t0 + 20 * 3600 + LATER)["audio_deleted"] == 1


def test_instances_sharing_data_dir(settings, speech_en, dialogue):  # I6
    from audio_mcp.context import build_app_context

    a = build_app_context(settings)
    b = build_app_context(settings.model_copy(deep=True))
    b.settings._instance_tag = "proc_%d_otherone" % __import__("os").getpid()  # model_copy keeps the tag
    fa, fb = a.settings.process_temp_dir / "x.bin", b.settings.process_temp_dir / "y.bin"
    fa.write_bytes(b"a")
    fb.write_bytes(b"b")
    assert fa.parent != fb.parent
    cleanup_temp(a.settings)  # startup cleanup: b is a live instance
    assert fb.exists() and fa.exists()
    assert cleanup_temp(a.settings, own=True) == 1 and not fa.exists() and fb.exists()
    dead = settings.resolved_temp_dir / "proc_999999999_deadbeef"
    dead.mkdir()
    (dead / "z.bin").write_bytes(b"z")
    assert cleanup_temp(a.settings) == 1 and not dead.exists() and fb.exists()

    ida, idb = _ingest(a, speech_en), _ingest(b, dialogue)
    a.settings.retention = "none"
    assert sweep(a, final=True)["audio_deleted"] == 1
    assert b.store.get(idb).audio_id == idb
    with pytest.raises(AudioMCPError):
        a.store.get(ida)


def test_sweep_keeps_old_results_of_live_audio(app, speech_en):
    import os

    aid = _ingest(app, speech_en)
    extract_audio_features(app, FeaturesRequest(source={"type": "audio_id", "audio_id": aid}))
    rdir = app.cache.root / aid
    old = time.time() - 3 * 24 * 3600
    os.utime(rdir, (old, old))  # results untouched for 3 days, but the audio is still in use
    orphan = app.cache.root / "aud_0000000000000000"
    orphan.mkdir()
    os.utime(orphan, (old, old))
    out = sweep(app)
    assert rdir.exists() and any(rdir.iterdir()) and not orphan.exists() and out["results_deleted"] == 1

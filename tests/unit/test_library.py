import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_mcp.tools.handlers import search_audio_library
from audio_mcp.tools.schemas import SearchLibraryRequest


def lib(app, **kw):
    return search_audio_library(app, SearchLibraryRequest(**kw))


def test_index_and_keyword_search(app, speech_en, dialogue):
    out = lib(app, action="index", paths=[str(speech_en), str(dialogue)])["result"]
    assert len(out["indexed"]) == 2 and out["failed"] == []
    m = lib(app, action="search", query="report")["result"]["matches"]
    assert m and "combined" in m[0]["scores"] and m[0]["start"] < m[0]["end"] and "report" in m[0]["excerpt"].lower()
    assert m[0]["source_available"] is True


def test_reindex_replaces(app, speech_en):
    lib(app, action="index", paths=[str(speech_en)])
    lib(app, action="index", paths=[str(speech_en)])
    assert len(lib(app, action="list")["result"]["recordings"]) == 1


def test_event_filter(app, synth, speech_en):
    lib(app, action="index", paths=[str(synth["alarm"]), str(speech_en)])
    m = lib(app, action="search", query="beep", filters={"event_label": "Beep, bleep"})["result"]["matches"]
    assert m and all(x["path"].endswith("alarm.wav") for x in m)


def test_folder_index(app, tmp_path, speech_en):
    d = tmp_path / "lib"; d.mkdir(); shutil.copy(speech_en, d / "a.wav"); (d / "notes.txt").write_text("x")
    out = lib(app, action="index", paths=[str(d)], async_mode="never")["result"]
    assert [Path(i["path"]).name for i in out["indexed"]] == ["a.wav"]


def test_missing_source_flag(app, tmp_path, speech_en):
    p = tmp_path / "copy.wav"; shutil.copy(speech_en, p)
    lib(app, action="index", paths=[str(p)]); p.unlink()
    m = lib(app, action="search", query="report")["result"]["matches"]
    assert any(x["source_available"] is False for x in m)


def test_remove(app, speech_en):
    aid = lib(app, action="index", paths=[str(speech_en)])["result"]["indexed"][0]["audio_id"]
    assert lib(app, action="remove", audio_id=aid)["result"]["removed"] is True
    assert lib(app, action="search", query="report")["result"]["matches"] == []


def test_query_by_example(app, synth, speech_en):
    lib(app, action="index", paths=[str(synth["alarm"]), str(speech_en)])
    m = lib(app, action="search", query_audio={"type": "file_path", "path": str(synth["alarm"])})["result"]["matches"]
    assert m[0]["path"].endswith("alarm.wav") and "audio_cosine" in m[0]["scores"]


def test_index_rejects_outside_root(app):
    r = lib(app, action="index", paths=[str(Path.home().parent / "nope.wav")])
    assert r["result"]["failed"][0]["error"]["code"] in {"PATH_NOT_ALLOWED", "FILE_NOT_FOUND"}


def test_validation():
    with pytest.raises(ValidationError):
        SearchLibraryRequest(action="search")
    with pytest.raises(ValidationError):
        SearchLibraryRequest(action="remove")

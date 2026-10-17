import base64
import hashlib
import re

import pytest

from audio_mcp.envelope import AudioMCPError
from audio_mcp.ingest.sources import AudioIdSource, Base64Source, FilePathSource
from audio_mcp.store.audio_store import AudioStore
from audio_mcp.store.cache import ResultCache, config_hash


def test_same_bytes_same_id_across_source_types(settings, speech_en):
    store = AudioStore(settings)
    a = store.ingest(FilePathSource(type="file_path", path=str(speech_en)))
    b = store.ingest(Base64Source(type="base64", data=base64.b64encode(speech_en.read_bytes()).decode(), filename_hint="x.wav"))
    assert a.audio_id == b.audio_id and re.fullmatch(r"aud_[0-9a-f]{16}", a.audio_id)
    assert not any(settings.resolved_temp_dir.iterdir())
    assert b.ingested_at == a.ingested_at and b.original_name == a.original_name


def test_original_untouched(settings, speech_en):
    before = hashlib.sha256(speech_en.read_bytes()).hexdigest()
    AudioStore(settings).ingest(FilePathSource(type="file_path", path=str(speech_en)))
    assert hashlib.sha256(speech_en.read_bytes()).hexdigest() == before


def test_unknown_id(settings):
    with pytest.raises(AudioMCPError) as e:
        AudioStore(settings).get("aud_0000000000000000")
    assert e.value.code == "AUDIO_NOT_FOUND"


def test_bad_file_rejected_and_nothing_stored(settings):
    store = AudioStore(settings)
    bad = Base64Source(type="base64", data=base64.b64encode(b"not audio at all" * 50).decode(), filename_hint="x.wav")
    with pytest.raises(AudioMCPError):
        store.ingest(bad)
    assert store.list_ids() == [] and not any(settings.resolved_temp_dir.iterdir())


def test_get_load_variant_delete_and_lru(settings, speech_en, dialogue):
    store = AudioStore(settings)
    a = store.ingest(FilePathSource(type="file_path", path=str(speech_en)))
    assert store.ingest(AudioIdSource(type="audio_id", audio_id=a.audio_id)).probe == a.probe
    audio = store.load(a.audio_id)
    assert store.load(a.audio_id) is audio
    v = store.put_variant(a.audio_id, "denoised", audio.native, audio.native_sr, [{"op": "x"}])
    assert v.exists() and v.with_suffix(".json").exists()
    b = store.ingest(FilePathSource(type="file_path", path=str(dialogue)))
    store.load(b.audio_id)
    assert store.list_ids() == sorted([a.audio_id, b.audio_id])
    store.delete(a.audio_id)
    assert store.list_ids() == [b.audio_id]
    with pytest.raises(AudioMCPError):
        store.load(a.audio_id)


def test_cache_roundtrip(settings):
    c = ResultCache(settings)
    h = config_hash({"m": "small"})
    assert c.get("aud_0000000000000000", "transcription", h) is None
    c.put("aud_0000000000000000", "transcription", h, {"text": "x"})
    assert c.get("aud_0000000000000000", "transcription", h) == {"text": "x"}
    c.clear("aud_0000000000000000")
    assert c.get("aud_0000000000000000", "transcription", h) is None


def test_concurrent_ingest_same_file(settings, speech_en):  # I5
    import threading

    store, ids, errs = AudioStore(settings), [], []
    gate = threading.Barrier(6)

    def go():
        gate.wait()
        try:
            ids.append(store.ingest(FilePathSource(type="file_path", path=str(speech_en))).audio_id)
        except Exception as e:  # noqa: BLE001 - the test reports any failure
            errs.append(repr(e))

    ts = [threading.Thread(target=go) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert errs == [] and len(ids) == 6 and len(set(ids)) == 1
    assert not list(store._dir(ids[0]).glob("*.tmp"))

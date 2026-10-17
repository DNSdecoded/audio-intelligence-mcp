import numpy as np
import soundfile as sf

from conftest import fp
from audio_mcp.backends.base import group_chunks
from audio_mcp.tools.handlers import transcribe_audio
from audio_mcp.tools.schemas import TranscribeRequest
from audio_mcp.types import Interval


def test_group_chunks():
    r = [Interval(start=0, end=1), Interval(start=1.5, end=2), Interval(start=3, end=4), Interval(start=4.2, end=40)]
    g = group_chunks(r)
    assert [(c.start, c.end) for c in g] == [(0, 2), (3, 4), (4.2, 34.2), (34.2, 40)]


def _two(speech_en, tmp_path):
    x, sr = sf.read(speech_en, dtype="float32")
    p = tmp_path / "two.wav"
    sf.write(p, np.concatenate([x, np.zeros(sr, dtype="float32"), x]), sr)
    return p


def test_language_runs_and_translation(app, tmp_path, speech_en):
    r = transcribe_audio(app, TranscribeRequest(source=fp(_two(speech_en, tmp_path)), multilingual=True,
                                                translate_to_en=True))
    res = r["result"]
    assert [l["language"] for l in res["languages"]] == ["en", "es"]
    assert res["segments"][0]["language"] == "en" and "translation" not in res["segments"][0]
    es = [s for s in res["segments"] if s["language"] == "es"]
    assert es and all(s["translation"] for s in es)
    assert any(w["code"] == "TRANSLATION_TARGET_LIMITED" for w in r["warnings"])
    assert res["code_switching"] is True


def test_monolingual_unchanged(app, speech_en):
    res = transcribe_audio(app, TranscribeRequest(source=fp(speech_en)))["result"]
    assert [l["language"] for l in res["languages"]] == ["en"] and res["code_switching"] is False

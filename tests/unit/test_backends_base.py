import sys

import numpy as np
import pytest
import soundfile as sf

from audio_mcp.backends.base import UnavailableBackend, build_backends
from audio_mcp.backends.energy_vad import EnergyVAD
from audio_mcp.backends.fake import fake_backends
from audio_mcp.envelope import AudioMCPError, ErrorCode
from audio_mcp.ingest.decode import resample
from audio_mcp.types import Interval


def test_energy_vad_finds_beeps(synth):
    x, _ = sf.read(synth["alarm"], dtype="float32")
    r = EnergyVAD().speech_regions(resample(x, 32000, 16000))
    assert r and r[0].start == pytest.approx(1.0, abs=0.1) and r[-1].end == pytest.approx(3.0, abs=0.15)


def test_energy_vad_silence(synth):
    x, _ = sf.read(synth["silence"], dtype="float32")
    assert EnergyVAD().speech_regions(x) == []


def test_real_profile_degrades_without_models(settings, monkeypatch):
    settings.backend_profile = "real"
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    b = build_backends(settings)
    ok, reason = b.asr.available()
    assert ok is False and "faster_whisper" in reason
    assert b.diarizer is None and b.vad_fallback.available()[0]
    with pytest.raises(AudioMCPError) as e:
        b.asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert e.value.code == ErrorCode.MODEL_UNAVAILABLE


def test_unavailable_backend_load_raises():
    with pytest.raises(AudioMCPError):
        UnavailableBackend("x", "why").load()


def test_fake_tagger_shape(speech_en):
    x, _ = sf.read(speech_en, dtype="float32")
    t = fake_backends().tagger.tag_windows(x)
    assert t.scores.shape[1] == 527 == len(t.labels)
    assert t.embeddings.shape == (len(t.scores), 2048)


def test_fake_tagger_labels_beep_and_silence(synth):
    tg = fake_backends().tagger
    x, _ = sf.read(synth["silence"], dtype="float32")
    assert tg.labels[int(tg.tag_windows(x).scores[0].argmax())] == "Silence"
    a, _ = sf.read(synth["alarm"], dtype="float32")
    t = tg.tag_windows(resample(a, 32000, 16000))
    assert tg.labels[int(t.scores[3].argmax())] == "Beep, bleep"  # window 1.5-2.5 s


def test_fake_asr_chunks_language_translate():
    b = fake_backends()
    r = b.asr.transcribe(np.zeros(16000, dtype=np.float32), per_chunk_language=True, task="translate",
                         chunks=[Interval(start=0, end=1), Interval(start=1, end=2)])
    assert [s.language for s in r.segments] == ["en", "es"]
    assert r.segments[0].text.startswith("[en] ")
    assert r.segments[0].id == "seg_0001"
    one = b.asr.transcribe(np.zeros(32000, dtype=np.float32), chunks=[Interval(start=0, end=1)])
    assert one.languages == [{"language": "en", "start": 0.0, "end": 2.0, "probability": 0.9}]
    assert r.segments[0].words[0].confidence_kind == "model_probability"


def test_fake_text_and_speaker():
    b = fake_backends()
    e = b.text.embed(["hello world", ""])
    assert e.shape == (2, 256) and np.linalg.norm(e[0]) == pytest.approx(1.0)
    assert b.speaker.embed([np.random.default_rng(0).standard_normal(16000)]).shape == (1, 64)


def test_real_profile_does_not_import_torch(settings):
    if "torch" in sys.modules:
        pytest.skip("torch already imported")
    settings.backend_profile = "real"
    build_backends(settings)
    assert "torch" not in sys.modules

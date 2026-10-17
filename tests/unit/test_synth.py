import shutil

import soundfile as sf

from conftest import FIXTURES_DIR


def test_synth_set(synth):
    info = sf.info(synth["sine"])
    assert info.samplerate == 44100 and info.channels == 2 and abs(info.duration - 2.0) < 1e-3
    assert sf.info(synth["hires"]).subtype == "PCM_24" and synth["empty"].stat().st_size == 0


def test_committed_speech_present():
    for n in ["speech_en.wav", "speech_en.txt", "dialogue_2spk.wav", "dialogue_2spk.rttm", "speech_es.ogg", "SOURCES.md"]:
        assert (FIXTURES_DIR / n).exists(), n
    assert sf.info(FIXTURES_DIR / "speech_en.wav").samplerate == 16000


def test_synth_contract_names(synth):
    names = "sine silence noise clipped alarm impulses spliced hum low_level dc_offset multichannel hires short corrupt not_audio empty".split()
    if shutil.which("ffmpeg"):
        names += [f"speech_en.{e}" for e in "flac mp3 ogg m4a webm aiff".split()]
    missing = [n for n in names if n not in synth or not synth[n].exists()]
    assert not missing

import pytest

from audio_mcp.dsp.speech import speech_characteristics


def w(s, e, t, spk=None):
    d = {"start": s, "end": e, "text": t, "confidence": .9, "confidence_kind": "model_probability"}
    return {**d, "speaker_id": spk} if spk else d


def test_wpm():
    res = {"transcription": {"segments": [{"id": "seg_0001", "start": 0, "end": 6, "text": "",
                                           "words": [w(i * .5, i * .5 + .4, "x") for i in range(12)]}]},
           "vad": {"speech_regions": [{"start": 0, "end": 6}]}, "features": {"values": {}}}
    s = speech_characteristics(res)
    assert s["words_per_minute"] == pytest.approx(120, abs=1) and s["pause_count"] == 0
    assert "not inferred" in s["notes"]


def test_pauses_and_speakers():
    words = [w(0, .4, "a", "speaker_1"), w(.5, .9, "b", "speaker_1"), w(2.5, 2.9, "c", "speaker_2"),
             w(3.2, 3.6, "d", "speaker_2")]
    res = {"speaker_assign": {"segments": [{"id": "seg_0001", "start": 0, "end": 3.6, "text": "", "words": words}]},
           "transcription": {"segments": []}, "vad": {"speech_regions": [{"start": 0, "end": 3.6}]},
           "features": {"values": {"f0_min_hz": 100.0, "f0_max_hz": 180.0, "f0_std_hz": 12.0}}}
    s = speech_characteristics(res)
    assert s["pause_count"] == 2 and s["long_pauses"] == [{"start": 0.9, "end": 2.5}]
    assert s["f0_range_hz"] == 80.0 and set(s["per_speaker"]) == {"speaker_1", "speaker_2"}
    assert s["articulation_rate_wpm"] > s["words_per_minute"]


def test_no_words():
    s = speech_characteristics({"transcription": {"segments": []}, "vad": {"speech_regions": []}, "features": {"values": {}}})
    assert s["words_per_minute"] is None and s["pause_count"] == 0

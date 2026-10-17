import pytest
from pydantic import ValidationError

from audio_mcp.types import Evidence, Event, Interval, Segment, SpeakerTurn, Word, r3


def test_segment_rounds_and_omits_none():
    d = Segment(id="seg_0001", start=1.23456, end=2.0, text="hi").model_dump(exclude_none=True)
    assert d["start"] == 1.235 and "confidence" not in d and "speaker_id" not in d


def test_r3_rounds_to_three_decimals():
    assert r3(1.23456) == 1.235
    assert isinstance(r3(2), float)


def test_all_time_fields_are_rounded():
    assert Interval(start=0.0004, end=1.0009).model_dump() == {"start": 0.0, "end": 1.001}
    assert Word(start=0.1111, end=0.2229, text="a").start == 0.111
    assert Event(id="evt_0001", label="x", start=1.0005, end=2.0, source="s").start == 1.0
    assert SpeakerTurn(speaker_id="speaker_1", start=0.5555, end=0.6666).end == 0.667
    assert Evidence(id="seg_0001", start=3.0004, end=4.0, source="asr").start == 3.0


def test_confidence_requires_kind():
    with pytest.raises(ValidationError):
        Word(start=0, end=1, text="a", confidence=0.9)
    assert Word(start=0, end=1, text="a", confidence=0.9, confidence_kind="heuristic").confidence == 0.9


def test_confidence_kind_is_restricted():
    with pytest.raises(ValidationError):
        Word(start=0, end=1, text="a", confidence=0.9, confidence_kind="calibrated")


def test_evidence_kind_defaults_to_detected():
    assert Evidence(id="evt_0001", start=0, end=1, source="panns").kind == "detected"
    with pytest.raises(ValidationError):
        Evidence(id="evt_0001", start=0, end=1, source="panns", kind="guess")


def test_segment_nests_words_and_omits_none_in_nested_dump():
    seg = Segment(id="seg_0001", start=0, end=1, text="a", words=[Word(start=0, end=0.5, text="a")])
    d = seg.model_dump(exclude_none=True)
    assert d["words"] == [{"start": 0.0, "end": 0.5, "text": "a"}]

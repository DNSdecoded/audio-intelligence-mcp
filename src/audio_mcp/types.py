from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

ConfidenceKind = Literal["model_probability", "heuristic"]


def r3(x: float) -> float:
    return round(float(x), 3)


class _Timed(BaseModel):
    @field_validator("start", "end", check_fields=False)
    @classmethod
    def _round_time(cls, v: float) -> float:
        return r3(v)


class _Scored(_Timed):
    # confidence is only set when a model or rule produced it, so it must carry its kind
    confidence: float | None = None
    confidence_kind: ConfidenceKind | None = None

    @model_validator(mode="after")
    def _kind_required_with_confidence(self):
        if self.confidence is not None and self.confidence_kind is None:
            raise ValueError("confidence requires confidence_kind")
        return self


class Interval(_Timed):
    start: float
    end: float


class Word(_Scored):
    start: float
    end: float
    text: str


class Segment(_Scored):
    id: str
    start: float
    end: float
    text: str
    language: str | None = None
    speaker_id: str | None = None
    words: list[Word] = []
    translation: str | None = None


class Event(_Scored):
    id: str
    label: str
    start: float
    end: float
    source: str


class SpeakerTurn(_Timed):
    speaker_id: str
    start: float
    end: float


class Evidence(_Scored):
    id: str
    start: float
    end: float
    source: str
    text: str | None = None
    label: str | None = None
    kind: Literal["detected", "interpretation"] = "detected"

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_mcp.analysis.events import EVENT_CATEGORIES
from audio_mcp.dsp.features import FEATURE_GROUPS
from audio_mcp.dsp.preprocess import PREPROCESS_OPS
from audio_mcp.ingest.sources import AudioSource
from audio_mcp.pipeline.planner import TASK_TO_STAGES

TaskName = Literal[
    "acoustic_features", "transcription", "language_detection", "sound_event_detection", "audio_quality",
    "speaker_diarization", "speech_characteristics", "integrity", "timeline", "summarization", "uncertainty",
]
CategoryName = Literal["alarm", "impact", "footsteps", "door", "animal", "machinery", "music", "environment"]
FeatureGroup = Literal["level", "silence", "spectral", "pitch", "rhythm", "bands"]
AsyncMode = Literal["auto", "always", "never"]
NodeType = Literal["speech_segment", "speaker_turn", "sound_event", "silence", "overlap", "transition"]
PreprocessOp = Literal["remove_dc", "mono", "highpass_80hz", "denoise", "trim_silence", "resample_16k", "normalize"]

# keep the Literals in step with the engine's tables
assert set(get_args(TaskName)) == set(TASK_TO_STAGES)
assert set(get_args(FeatureGroup)) == set(FEATURE_GROUPS)
assert set(get_args(PreprocessOp)) == set(PREPROCESS_OPS)
assert set(get_args(CategoryName)) <= set(EVENT_CATEGORIES)


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeOptions(_Req):
    timestamps: bool = True
    word_level_timestamps: bool = True
    identify_speakers: bool = True
    detect_emotions: bool = False
    include_confidence: bool = True
    include_evidence: bool = True
    language: str | None = None
    num_speakers: int | None = Field(None, ge=1, le=20)


class AnalyzeRequest(_Req):
    source: AudioSource
    tasks: list[TaskName] | Literal["auto"] = "auto"
    goal: str | None = None
    budget: Literal["fast", "balanced", "thorough"] = "balanced"
    options: AnalyzeOptions = AnalyzeOptions()
    output_format: Literal["json"] = "json"
    async_mode: AsyncMode = "auto"


class TranscribeRequest(_Req):
    source: AudioSource
    language: str | None = None
    word_timestamps: bool = True
    diarize: bool = False
    translate_to_en: bool = False
    multilingual: bool = False
    async_mode: AsyncMode = "auto"


class DetectEventsRequest(_Req):
    source: AudioSource
    categories: list[CategoryName] | None = None
    labels: list[str] | None = None
    min_confidence: float = Field(0.3, ge=0, le=1)
    async_mode: AsyncMode = "auto"


class FeaturesRequest(_Req):
    source: AudioSource
    feature_groups: list[FeatureGroup] | None = None
    async_mode: AsyncMode = "auto"


class InspectQualityRequest(_Req):
    source: AudioSource
    apply_preprocessing: bool | list[PreprocessOp] = False
    compare_transcription: bool = False


class TimelineRequest(_Req):
    source: AudioSource
    include: list[NodeType] | None = None
    diarize: bool = True


class AskRequest(_Req):
    source: AudioSource
    question: str = Field(min_length=3, max_length=1000)
    include_evidence: bool = True
    use_llm: bool | None = None


class SummarizeRequest(_Req):
    source: AudioSource
    length: Literal["short", "medium", "long"] = "short"
    focus: str | None = Field(None, max_length=500)
    use_llm: bool | None = None


class CompareRequest(_Req):
    sources: list[AudioSource] = Field(min_length=1, max_length=10)
    baseline: list[AudioSource] | None = Field(None, min_length=1, max_length=20)
    aspects: list[Literal["features", "events", "transcript", "embedding"]] | None = None
    z_threshold: float = Field(3.0, gt=0)

    @model_validator(mode="after")
    def _two_or_baseline(self):
        if not self.baseline and len(self.sources) < 2:
            raise ValueError("compare needs at least 2 sources, or 1+ sources and a baseline")
        return self


class LibraryFilters(_Req):
    language: str | None = None
    min_duration: float | None = Field(None, ge=0)
    max_duration: float | None = Field(None, ge=0)
    event_label: str | None = None
    speaker_count: int | None = Field(None, ge=1)
    audio_ids: list[str] | None = None


class SearchLibraryRequest(_Req):
    action: Literal["index", "search", "list", "remove"]
    paths: list[str] | None = Field(None, min_length=1, max_length=1000)
    query: str | None = Field(None, max_length=1000)
    query_audio: AudioSource | None = None
    filters: LibraryFilters | None = None
    limit: int = Field(10, ge=1, le=100)
    audio_id: str | None = Field(None, pattern=r"^aud_[0-9a-f]{16}$")
    async_mode: AsyncMode = "auto"

    @model_validator(mode="after")
    def _per_action(self):
        need = {"index": self.paths, "remove": self.audio_id}.get(self.action, True)
        if self.action == "search":
            need = self.query or self.query_audio
        if not need:
            raise ValueError({"index": "index needs paths", "remove": "remove needs audio_id",
                              "search": "search needs query or query_audio"}[self.action])
        return self


class GetJobRequest(_Req):
    job_id: str = Field(pattern=r"^job_[0-9a-f]{12}$")
    action: Literal["status", "cancel"] = "status"

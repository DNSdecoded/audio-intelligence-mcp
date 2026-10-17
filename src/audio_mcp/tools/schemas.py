from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

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

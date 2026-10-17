"""Backend protocols, result types and the registry."""
import importlib.util
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.types import Interval, Segment, SpeakerTurn


class Backend(Protocol):
    info: ModelInfo

    def available(self) -> tuple[bool, str | None]: ...  # cheap import check
    def load(self) -> None: ...  # idempotent; raises MODEL_UNAVAILABLE


class VAD(Backend, Protocol):
    def speech_regions(self, mono16k: np.ndarray) -> list[Interval]: ...


@dataclass
class ASRResult:
    segments: list[Segment]
    language: str | None
    language_probability: float | None
    languages: list[dict] = field(default_factory=list)
    dropped_low_speech: int = 0


class ASR(Backend, Protocol):
    def transcribe(
        self,
        mono16k: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
        task: Literal["transcribe", "translate"] = "transcribe",
        per_chunk_language: bool = False,
        chunks: list[Interval] | None = None,
        **decode_kwargs,
    ) -> ASRResult: ...


@dataclass
class TagResult:
    labels: list[str]
    scores: np.ndarray  # windows x 527
    window_starts: np.ndarray
    window_seconds: float
    embeddings: np.ndarray


class Tagger(Backend, Protocol):
    def tag_windows(self, mono16k: np.ndarray) -> TagResult: ...
    def embed(self, mono16k: np.ndarray) -> np.ndarray: ...


class SpeakerEmbedder(Backend, Protocol):
    def embed(self, windows: list[np.ndarray]) -> np.ndarray: ...  # (n, d)


class Diarizer(Backend, Protocol):
    def diarize(
        self, mono16k: np.ndarray, num_speakers: int | None
    ) -> tuple[list[SpeakerTurn], list[Interval]]: ...


class TextEmbedder(Backend, Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...  # L2-normalized


class LLM(Backend, Protocol):
    is_remote: bool

    def complete(self, system: str, prompt: str) -> str: ...


@dataclass
class UnavailableBackend:
    name: str
    reason: str
    is_remote: bool = False

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(stage="unavailable", name=self.name, version="0")

    def available(self) -> tuple[bool, str | None]:
        return False, self.reason

    def _fail(self, *_, **__):
        raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, f"{self.name} unavailable: {self.reason}")

    load = _fail

    def __getattr__(self, attr):  # any capability method (transcribe, embed, ...) fails the same way
        if attr.startswith("_"):
            raise AttributeError(attr)
        return self._fail


@dataclass
class Backends:
    vad: object
    vad_fallback: object
    asr: object
    tagger: object
    speaker: object
    diarizer: object | None
    text: object
    llm: object


def _importable(module: str) -> bool:
    try:  # find_spec: no import, so torch stays unloaded until a model is used
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):  # ValueError: sys.modules[name] is None
        return False


def build_backends(settings) -> Backends:
    from audio_mcp.backends.energy_vad import EnergyVAD
    from audio_mcp.backends.fake import NoLLM, fake_backends

    if settings.backend_profile == "fake":
        return fake_backends()

    hint = "not installed (uv sync --extra models)"
    todo = "not implemented yet"  # later tasks replace these with real adapters
    if _importable("silero_vad"):
        from audio_mcp.backends.silero import SileroVAD

        vad = SileroVAD()
    else:
        vad = UnavailableBackend("silero-vad", f"silero_vad {hint}")
    if _importable("faster_whisper"):
        from audio_mcp.backends.whisper import FasterWhisperASR

        asr = FasterWhisperASR(settings)
    else:
        asr = UnavailableBackend("faster-whisper", f"faster_whisper {hint}")
    return Backends(
        vad=vad,
        vad_fallback=EnergyVAD(),
        asr=asr,
        tagger=UnavailableBackend("panns", todo),
        speaker=UnavailableBackend("ecapa", todo),
        diarizer=None,
        text=UnavailableBackend("sentence-transformers", todo),
        llm=NoLLM(),
    )

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from audio_mcp.config import load_settings
from audio_mcp.context import AppContext, build_app_context
from audio_mcp.ingest.sources import AudioSource
from audio_mcp.logging_setup import setup_logging
from audio_mcp.store.retention import cleanup_temp, sweep
from audio_mcp.tools import handlers
from audio_mcp.tools.schemas import (
    AnalyzeOptions, AnalyzeRequest, AskRequest, AsyncMode, CompareRequest, GetJobRequest, CategoryName, DetectEventsRequest, FeatureGroup, FeaturesRequest,
    InspectQualityRequest, LibraryFilters, NodeType, SearchLibraryRequest, StreamRequest, SummarizeRequest, PreprocessOp, TaskName, TimelineRequest, TranscribeRequest,
)

_app: AppContext | None = None


@asynccontextmanager
async def lifespan(_server):
    global _app
    settings = load_settings()
    setup_logging(settings)
    _app = build_app_context(settings)
    log = logging.getLogger("audio_mcp")
    log.info("server started; temp files removed: %d; sweep: %s", cleanup_temp(settings), sweep(_app))

    stop = threading.Event()

    def hourly():
        while not stop.wait(3600):
            log.info("retention sweep: %s", sweep(_app))

    threading.Thread(target=hourly, name="audio-mcp-retention", daemon=True).start()
    try:
        yield
    finally:
        stop.set()
        _app.jobs.shutdown()
        sweep(_app, final=True)
        cleanup_temp(settings)


mcp = FastMCP("audio-intelligence", lifespan=lifespan)


async def _run(handler, model, params: dict) -> dict[str, Any]:
    try:
        req = model(**params)
    except ValidationError as e:
        return handlers.invalid_input(e)
    return await anyio.to_thread.run_sync(lambda: handler(_app, req))


@mcp.tool(name="analyze_audio", description=(
    "Run a multi-task analysis of an audio source (transcription, language, sound events, acoustic features). "
    "Returns a JSON envelope with an `analysis` object plus warnings, errors and processing info."))
async def analyze_audio(
    source: AudioSource,
    tasks: list[TaskName] | Literal["auto"] = "auto",
    goal: str | None = None,
    budget: Literal["fast", "balanced", "thorough"] = "balanced",
    options: AnalyzeOptions | None = None,
    output_format: Literal["json"] = "json",
    async_mode: AsyncMode = "auto",
) -> dict:
    p = dict(source=source, tasks=tasks, goal=goal, budget=budget, output_format=output_format, async_mode=async_mode)
    if options is not None:
        p["options"] = options
    return await _run(handlers.analyze_audio, AnalyzeRequest, p)


@mcp.tool(name="transcribe_audio", description=(
    "Transcribe speech in an audio source with segment and optional word timestamps. "
    "Returns text, segments, detected language and per-chunk languages."))
async def transcribe_audio(
    source: AudioSource,
    language: str | None = None,
    word_timestamps: bool = True,
    diarize: bool = False,
    translate_to_en: bool = False,
    multilingual: bool = False,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.transcribe_audio, TranscribeRequest, dict(
        source=source, language=language, word_timestamps=word_timestamps, diarize=diarize,
        translate_to_en=translate_to_en, multilingual=multilingual, async_mode=async_mode))


@mcp.tool(name="detect_audio_events", description=(
    "Detect sound events (alarms, impacts, music, animals, ...) in an audio source, optionally filtered by "
    "category or label. Returns timestamped events with model-probability confidence."))
async def detect_audio_events(
    source: AudioSource,
    categories: list[CategoryName] | None = None,
    labels: list[str] | None = None,
    min_confidence: float = 0.3,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.detect_audio_events, DetectEventsRequest, dict(
        source=source, categories=categories, labels=labels, min_confidence=min_confidence, async_mode=async_mode))


@mcp.tool(name="extract_audio_features", description=(
    "Compute acoustic features (level, silence, spectral, pitch, rhythm, bands) for an audio source. "
    "Returns the audio metadata and the requested feature groups."))
async def extract_audio_features(
    source: AudioSource,
    feature_groups: list[FeatureGroup] | None = None,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.extract_audio_features, FeaturesRequest, dict(
        source=source, feature_groups=feature_groups, async_mode=async_mode))


@mcp.tool(name="inspect_audio_quality", description=(
    "Diagnose recording quality (clipping, low level, silence, noise, hum, echo, bandwidth, DC offset) and "
    "optionally write a preprocessed variant; the original is never modified. Returns issues, metrics and "
    "recommended preprocessing."))
async def inspect_audio_quality(
    source: AudioSource,
    apply_preprocessing: bool | list[PreprocessOp] = False,
    compare_transcription: bool = False,
) -> dict:
    return await _run(handlers.inspect_audio_quality, InspectQualityRequest, dict(
        source=source, apply_preprocessing=apply_preprocessing, compare_transcription=compare_transcription))


@mcp.tool(name="get_audio_timeline", description=(
    "Build a chronological event graph of an audio source: speech segments, speaker turns, sound events, silence, "
    "overlaps and transitions, with relation edges (precedes, during, overlaps, interrupts) and a text rendering."))
async def get_audio_timeline(
    source: AudioSource,
    include: list[NodeType] | None = None,
    diarize: bool = True,
) -> dict:
    return await _run(handlers.get_audio_timeline, TimelineRequest, dict(source=source, include=include, diarize=diarize))


@mcp.tool(name="ask_audio", description=(
    "Answer a natural-language question about an audio source (e.g. when something is mentioned, what happens "
    "before a sound, whether a sound is present) using timestamped evidence from the analysis. Says when the "
    "recording does not support a reliable answer."))
async def ask_audio(
    source: AudioSource,
    question: str,
    include_evidence: bool = True,
    use_llm: bool | None = None,
) -> dict:
    return await _run(handlers.ask_audio, AskRequest, dict(
        source=source, question=question, include_evidence=include_evidence, use_llm=use_llm))


@mcp.tool(name="summarize_audio", description=(
    "Summarize an audio source extractively: key sentences, topics and key moments, each linked to transcript "
    "segment or event ids. Non-speech audio is summarized from detected sounds and features."))
async def summarize_audio(
    source: AudioSource,
    length: Literal["short", "medium", "long"] = "short",
    focus: str | None = None,
    use_llm: bool | None = None,
) -> dict:
    return await _run(handlers.summarize_audio, SummarizeRequest, dict(
        source=source, length=length, focus=focus, use_llm=use_llm))


@mcp.tool(name="compare_audio", description=(
    "Compare two or more recordings (feature, band, event, transcript and embedding differences), or compare "
    "recordings against baseline recordings to localize anomalous windows with z-scores. Measured differences are "
    "kept separate from hedged interpretations."))
async def compare_audio(
    sources: list[AudioSource],
    baseline: list[AudioSource] | None = None,
    aspects: list[Literal["features", "events", "transcript", "embedding"]] | None = None,
    z_threshold: float = 3.0,
) -> dict:
    return await _run(handlers.compare_audio, CompareRequest, dict(
        sources=sources, baseline=baseline, aspects=aspects, z_threshold=z_threshold))


@mcp.tool(name="search_audio_library", description=(
    "Index audio files/folders into a persistent local library (transcripts, sound events, embeddings) and search "
    "it by keywords, natural language, filters, or an example audio clip. Returns matching recordings and "
    "timestamped segments with scores."))
async def search_audio_library(
    action: Literal["index", "search", "list", "remove"],
    paths: list[str] | None = None,
    query: str | None = None,
    query_audio: AudioSource | None = None,
    filters: LibraryFilters | None = None,
    limit: int = 10,
    audio_id: str | None = None,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.search_audio_library, SearchLibraryRequest, dict(
        action=action, paths=paths, query=query, query_audio=query_audio, filters=filters, limit=limit,
        audio_id=audio_id, async_mode=async_mode))


@mcp.tool(name="get_job", description=(
    "Check the status/progress of an async analysis job (returned as job_id with status 'processing' by long "
    "analyses) or cancel it. When finished, the job's full response is included."))
async def get_job(job_id: str, action: Literal["status", "cancel"] = "status") -> dict:
    return await _run(handlers.get_job, GetJobRequest, dict(job_id=job_id, action=action))


@mcp.tool(name="analyze_audio_stream", description=(
    "Chunked streaming sessions (raw PCM or self-contained audio files per chunk): start a session, push chunks "
    "for provisional features/events/transcript, then finish for a full analysis plus a reconciliation of "
    "provisional vs final results. Live socket streaming and split encoded streams are not supported."))
async def analyze_audio_stream(
    action: Literal["start", "push", "status", "finish", "cancel"],
    session_id: str | None = None,
    chunk_format: Literal["pcm", "file"] | None = None,
    sample_rate_hz: int | None = None,
    channels: int | None = None,
    sample_format: Literal["s16le", "f32le"] | None = None,
    data: str | None = None,
    path: str | None = None,
    tasks: list[TaskName] | Literal["auto"] = "auto",
) -> dict:
    return await _run(handlers.analyze_audio_stream, StreamRequest, dict(
        action=action, session_id=session_id, chunk_format=chunk_format, sample_rate_hz=sample_rate_hz,
        channels=channels, sample_format=sample_format, data=data, path=path, tasks=tasks))


def main() -> None:
    mcp.run()

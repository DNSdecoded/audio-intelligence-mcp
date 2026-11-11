import hmac
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ValidationError
from starlette.responses import Response

from audio_mcp.config import load_settings
from audio_mcp.context import AppContext, build_app_context
from audio_mcp.ingest.sources import AudioSource
from audio_mcp.jobs import JOB_CONTEXT
from audio_mcp.logging_setup import setup_logging
from audio_mcp.store.retention import cleanup_temp, sweep
from audio_mcp.tools import handlers
from audio_mcp.tools.schemas import (
    AnalyzeOptions, AnalyzeRequest, AskRequest, AsyncMode, CompareRequest, ExportRequest, GetJobRequest, CategoryName, DetectEventsRequest, FeatureGroup, FeaturesRequest,
    InspectQualityRequest, LibraryFilters, NodeType, SearchLibraryRequest, StreamRequest, SummarizeRequest, PreprocessOp, TaskName, TimelineRequest, TranscribeRequest,
)

_app: AppContext | None = None
_refs = 0  # live lifespan entries; HTTP runs one per MCP session plus one for the whole server, stdio just one
_stop_app = None


def _start_app() -> None:
    global _app, _stop_app
    settings = load_settings()
    setup_logging(settings)
    # the stdio transport already holds the real stdout; stray library prints (panns_inference prints on load)
    # must not reach the JSON-RPC stream
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    _app = build_app_context(settings)
    log = logging.getLogger("audio_mcp")
    log.info("server started; temp files removed: %d; sweep: %s", cleanup_temp(settings), sweep(_app))

    stop = threading.Event()

    def hourly():
        while not stop.wait(3600):
            log.info("retention sweep: %s", sweep(_app))

    threading.Thread(target=hourly, name="audio-mcp-retention", daemon=True).start()

    def stop_app():
        global _app
        stop.set()
        _app.jobs.shutdown()
        sweep(_app, final=True)
        cleanup_temp(settings, own=True)
        sys.stdout = real_stdout
        _app = None

    _stop_app = stop_app


@asynccontextmanager
async def lifespan(_server):
    global _refs
    if _refs == 0:
        _start_app()
    _refs += 1
    try:
        yield
    finally:
        _refs -= 1
        if _refs == 0:
            _stop_app()


INSTRUCTIONS = """\
Local audio analysis. Every source is {"type": "file_path"|"base64"|"url"|"audio_id", ...}.
- Unsure what a recording holds: analyze_audio (tasks "auto", goal e.g. "meeting"). It returns an audio_id.
- Reuse that audio_id as the source for ask_audio, summarize_audio, get_audio_timeline and export_audio_analysis;
  stored audio expires after retention_ttl_hours (then re-send the file).
- Transcript/speakers/translation: transcribe_audio (hotwords for names and jargon). Sound events: detect_audio_events.
  Levels, noise, clipping: inspect_audio_quality / extract_audio_features.
- Differences from normal: compare_audio with baseline. Many files: search_audio_library (index, then search).
- status "processing" with a job_id means long audio: poll get_job.
- Read warnings and processing.stages_skipped before trusting a result; confidence values are uncalibrated.
"""

mcp = FastMCP("audio-intelligence", instructions=INSTRUCTIONS, lifespan=lifespan)


def _progress_reporter():
    """Callback for the runner that sends MCP progress notifications (no-op when the client sent no
    progressToken). The count of finished stages only grows, as the spec requires."""
    try:
        ctx = mcp.get_context()
        ctx.request_context  # raises outside a request
    except ValueError:
        return None
    sent = 0

    def report(_done, _total):
        nonlocal sent
        sent += 1
        try:
            anyio.from_thread.run(ctx.report_progress, sent)
        except Exception:  # progress is best effort; never fail the analysis over it
            pass

    return report


async def _run(handler, model, params: dict) -> dict[str, Any]:
    try:
        req = model(**params)
    except ValidationError as e:
        return handlers.invalid_input(e)
    progress = _progress_reporter()

    def call():
        if progress is not None:  # the runner reads this, as it does for async jobs
            JOB_CONTEXT.set((None, progress))
        return handler(_app, req)

    return await anyio.to_thread.run_sync(call)


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
    hotwords: list[str] | None = None,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.transcribe_audio, TranscribeRequest, dict(
        source=source, language=language, word_timestamps=word_timestamps, diarize=diarize,
        translate_to_en=translate_to_en, multilingual=multilingual, hotwords=hotwords, async_mode=async_mode))


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
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.inspect_audio_quality, InspectQualityRequest, dict(
        source=source, apply_preprocessing=apply_preprocessing, compare_transcription=compare_transcription,
        async_mode=async_mode))


@mcp.tool(name="get_audio_timeline", description=(
    "Build a chronological event graph of an audio source: speech segments, speaker turns, sound events, silence, "
    "overlaps and transitions, with relation edges (precedes, during, overlaps, interrupts) and a text rendering."))
async def get_audio_timeline(
    source: AudioSource,
    include: list[NodeType] | None = None,
    diarize: bool = True,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.get_audio_timeline, TimelineRequest, dict(
        source=source, include=include, diarize=diarize, async_mode=async_mode))


@mcp.tool(name="ask_audio", description=(
    "Answer a natural-language question about an audio source (e.g. when something is mentioned, what happens "
    "before a sound, whether a sound is present) using timestamped evidence from the analysis. Says when the "
    "recording does not support a reliable answer."))
async def ask_audio(
    source: AudioSource,
    question: str,
    include_evidence: bool = True,
    use_llm: bool | None = None,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.ask_audio, AskRequest, dict(
        source=source, question=question, include_evidence=include_evidence, use_llm=use_llm, async_mode=async_mode))


@mcp.tool(name="summarize_audio", description=(
    "Summarize an audio source extractively: key sentences, topics and key moments, each linked to transcript "
    "segment or event ids. Non-speech audio is summarized from detected sounds and features."))
async def summarize_audio(
    source: AudioSource,
    length: Literal["short", "medium", "long"] = "short",
    focus: str | None = None,
    use_llm: bool | None = None,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.summarize_audio, SummarizeRequest, dict(
        source=source, length=length, focus=focus, use_llm=use_llm, async_mode=async_mode))


@mcp.tool(name="compare_audio", description=(
    "Compare two or more recordings (feature, band, event, transcript and embedding differences), or compare "
    "recordings against baseline recordings to localize anomalous windows with z-scores. Measured differences are "
    "kept separate from hedged interpretations."))
async def compare_audio(
    sources: list[AudioSource],
    baseline: list[AudioSource] | None = None,
    aspects: list[Literal["features", "events", "transcript", "embedding"]] | None = None,
    z_threshold: float = 3.0,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.compare_audio, CompareRequest, dict(
        sources=sources, baseline=baseline, aspects=aspects, z_threshold=z_threshold, async_mode=async_mode))


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


@mcp.tool(name="export_audio_analysis", description=(
    "Export the analysis of a previously ingested audio_id to json, markdown, srt, vtt, csv or rttm in the "
    "configured export directory, optionally redacted. Returns the file path, size and sha256."))
async def export_audio_analysis(
    audio_id: str,
    format: Literal["json", "markdown", "srt", "vtt", "csv", "rttm"],
    tasks: list[TaskName] | Literal["auto"] = "auto",
    redact: bool = False,
    async_mode: AsyncMode = "auto",
) -> dict:
    return await _run(handlers.export_audio_analysis, ExportRequest, dict(
        audio_id=audio_id, format=format, tasks=tasks, redact=redact, async_mode=async_mode))


class _BearerAuth:
    """Pure ASGI middleware: every HTTP request needs `Authorization: Bearer <token>` (SSE streams stay unbuffered)."""

    def __init__(self, app, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                await Response("unauthorized", 401, headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
                return
        await self.app(scope, receive, send)


def http_app(token: str, host: str = "127.0.0.1"):
    """Streamable-HTTP ASGI app behind bearer auth. One token = one trust domain: every client holding it shares
    the audio store, result cache, jobs and library (no per-client isolation)."""
    if host not in ("127.0.0.1", "localhost", "::1"):  # FastMCP's Host allow-list only knows loopback
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = mcp.streamable_http_app()
    inner = app.router.lifespan_context

    @asynccontextmanager
    async def serve_lifespan(a):  # one app context for the server's life, not one per MCP session
        async with lifespan(None), inner(a):
            yield

    app.router.lifespan_context = serve_lifespan
    app.add_middleware(_BearerAuth, token=token)
    return app


def main() -> None:
    s = load_settings()
    if s.transport == "stdio":
        mcp.run()
        return
    import uvicorn

    token = os.environ.get(s.http_auth_token_env)
    if not token:
        raise SystemExit(f"HTTP transport needs a bearer token in ${s.http_auth_token_env}")
    if "allowed_roots" not in s.model_fields_set:
        raise SystemExit("HTTP transport needs allowed_roots set explicitly; the default (home directory) "
                         "would expose it to every client holding the token")
    uvicorn.run(http_app(token, s.http_host), host=s.http_host, port=s.http_port, log_level="warning")

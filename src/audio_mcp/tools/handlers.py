import functools
import logging

from pydantic import ValidationError

from audio_mcp.analysis.events import EVENT_CATEGORIES
from audio_mcp.envelope import AudioMCPError, ErrorCode, build_response, error_response, warning
from audio_mcp.pipeline.planner import TASK_TO_STAGES
from audio_mcp.pipeline.runner import run_analysis
from audio_mcp.tools.schemas import AnalyzeOptions, AnalyzeRequest, DetectEventsRequest, FeaturesRequest, TranscribeRequest

log = logging.getLogger("audio_mcp.tools")

PHASE1_AUTO = ["acoustic_features", "transcription", "language_detection", "sound_event_detection"]  # Task 17 replaces


def invalid_input(e: ValidationError) -> dict:
    err = e.errors()[0]
    loc = ".".join(str(x) for x in err["loc"])
    return error_response(AudioMCPError(ErrorCode.INVALID_INPUT, f"{loc}: {err['msg']}" if loc else err["msg"]))


def tool_handler(fn):
    @functools.wraps(fn)
    def wrapper(app, req):
        try:
            return fn(app, req)
        except AudioMCPError as e:
            return error_response(e)
        except ValidationError as e:
            return invalid_input(e)
        except Exception as e:  # log the type only, never data
            log.error("tool %s failed: %s", fn.__name__, type(e).__name__)
            return error_response(AudioMCPError(ErrorCode.STAGE_FAILED, f"{fn.__name__} failed: {type(e).__name__}"))

    return wrapper


def _strip(x, drop):
    if isinstance(x, dict):
        return {k: _strip(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, list):
        return [_strip(v, drop) for v in x]
    return x


def analysis_payload(outcome, options: AnalyzeOptions) -> dict:
    r = outcome.results
    a: dict = {}
    if "metadata" in r:
        a["audio_metadata"] = r["metadata"]
    if "language" in r:
        a["language"] = {k: r["language"][k] for k in ("primary", "confidence", "confidence_kind") if k in r["language"]}
    if "transcription" in r:
        a["transcription"] = {"text": r["transcription"]["text"], "segments": r["transcription"]["segments"]}
    if "events" in r:
        a["sound_events"] = r["events"]["events"]
    if "features" in r:
        a["acoustic_features"] = r["features"]
    if not options.word_level_timestamps:
        a = _strip(a, {"words"})
    if not options.include_confidence:
        a = _strip(a, {"confidence", "confidence_kind"})
    return a


def _respond(outcome, result, *, key="result", extra_warnings=()):
    return build_response(result=result, audio_id=outcome.record.audio_id, payload_key=key,
                          warnings=[*outcome.warnings, *extra_warnings], errors=outcome.errors,
                          processing=outcome.processing)


@tool_handler
def analyze_audio(app, req: AnalyzeRequest) -> dict:
    tasks = PHASE1_AUTO if req.tasks == "auto" else req.tasks
    stages = ["metadata"]  # always report audio_metadata
    for t in tasks:
        stages += [s for s in TASK_TO_STAGES[t] if s not in stages]
    o = req.options
    outcome = run_analysis(app, req.source, stages, {"language": o.language, "word_timestamps": o.word_level_timestamps})
    warns = [warning("UNSUPPORTED_FEATURE", "emotion detection is not implemented")] if o.detect_emotions else []
    return _respond(outcome, analysis_payload(outcome, o), key="analysis", extra_warnings=warns)


@tool_handler
def transcribe_audio(app, req: TranscribeRequest) -> dict:
    outcome = run_analysis(app, req.source, ["transcription"], {
        "language": req.language, "word_timestamps": req.word_timestamps, "multilingual": req.multilingual})
    warns = [warning("UNSUPPORTED_FEATURE", f"{n} is not implemented yet")
             for n, on in (("diarize", req.diarize), ("translate_to_en", req.translate_to_en)) if on]
    t = outcome.results.get("transcription")
    result = {k: t[k] for k in ("text", "segments", "language", "languages")} if t else None
    return _respond(outcome, result, extra_warnings=warns)


@tool_handler
def detect_audio_events(app, req: DetectEventsRequest) -> dict:
    labels = list(req.labels or [])
    for c in req.categories or []:
        labels += [x for x in EVENT_CATEGORIES[c] if x not in labels]
    outcome = run_analysis(app, req.source, ["events"], {
        "event_threshold": req.min_confidence, "event_labels": labels or None})
    ev = outcome.results.get("events")
    return _respond(outcome, {"events": ev["events"], "threshold": ev["threshold"]} if ev else None)


@tool_handler
def extract_audio_features(app, req: FeaturesRequest) -> dict:
    outcome = run_analysis(app, req.source, ["metadata", "features"], {"feature_groups": req.feature_groups})
    r = outcome.results
    return _respond(outcome, {"metadata": r["metadata"], "features": r["features"]} if "features" in r else None)

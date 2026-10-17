import functools
import logging

from pydantic import ValidationError

import numpy as np

from audio_mcp.analysis.events import EVENT_CATEGORIES
from audio_mcp.dsp.features import extract_features
from audio_mcp.dsp.preprocess import apply_preprocessing
from audio_mcp.ingest.decode import resample
from audio_mcp.envelope import AudioMCPError, ErrorCode, build_response, error_response, warning
from audio_mcp.pipeline.planner import TASK_TO_STAGES
from audio_mcp.pipeline.runner import run_analysis
from audio_mcp.tools.schemas import (
    AnalyzeOptions, AnalyzeRequest, DetectEventsRequest, FeaturesRequest, InspectQualityRequest, TranscribeRequest,
)

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
    if "quality" in r:
        a["audio_quality"] = r["quality"]
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


def _level_metrics(native, sr) -> dict:
    v = extract_features(native, sr, ["level", "silence"])["values"]
    return {k: v[k] for k in ("rms", "rms_dbfs", "peak_amplitude", "clipping_ratio", "dc_offset", "silence_ratio")}


def _asr_stats(segments, low: float) -> dict:
    conf = [w.confidence for s in segments for w in s.words if w.confidence is not None]
    return {"mean_word_confidence": round(float(np.mean(conf)), 3) if conf else None,
            "low_confidence_words": sum(c < low for c in conf), "words": len(conf)}


def compare_preprocessed_asr(app, record, audio, ops) -> dict:
    """Transcribe the original and a preprocessed variant; report measured confidence stats only."""
    asr = app.backends.asr
    ok, reason = asr.available()
    if not ok:
        return {"status": "skipped", "reason": reason}
    y, sr, transforms = apply_preprocessing(audio.native, audio.native_sr, ops)
    y16 = resample(y.mean(axis=0), sr, 16000) if sr != 16000 else y.mean(axis=0)
    offset = next((t["params"].get("offset_seconds", 0.0) for t in transforms if t["op"] == "trim_silence"), 0.0)
    low = app.settings.low_confidence_threshold
    before = asr.transcribe(audio.mono16k).segments
    after = asr.transcribe(y16).segments
    return {"before": _asr_stats(before, low), "after": _asr_stats(after, low), "time_offset_seconds": offset,
            "note": "measured comparison only; not a claim of improved accuracy"}


@tool_handler
def inspect_audio_quality(app, req: InspectQualityRequest) -> dict:
    outcome = run_analysis(app, req.source, ["quality"], {})
    q = outcome.results.get("quality")
    if q is None or req.apply_preprocessing is False:
        return _respond(outcome, q)
    ops = q["recommendations"] if req.apply_preprocessing is True else list(req.apply_preprocessing)
    result = dict(q)
    if not ops:
        result["preprocessing"] = {"applied": False, "reason": "no preprocessing recommended"}
        return _respond(outcome, result)
    audio = app.store.load(outcome.record.audio_id)
    y, sr, transforms = apply_preprocessing(audio.native, audio.native_sr, ops)
    path = app.store.put_variant(outcome.record.audio_id, "preprocessed", y, sr, transforms)
    pre = {"variant_path": str(path), "transforms": transforms,
           "before": _level_metrics(audio.native, audio.native_sr), "after": _level_metrics(y, sr)}
    if req.compare_transcription:
        pre["transcription_comparison"] = compare_preprocessed_asr(app, outcome.record, audio, ops)
    result["preprocessing"] = pre
    return _respond(outcome, result)

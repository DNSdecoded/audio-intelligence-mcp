import functools
import logging

from pydantic import ValidationError

import numpy as np

from audio_mcp.analysis.events import EVENT_CATEGORIES
from audio_mcp.dsp.features import extract_features
from audio_mcp.dsp.preprocess import apply_preprocessing
from audio_mcp.ingest.decode import resample
from audio_mcp.envelope import AudioMCPError, ErrorCode, build_response, error_response, warning
from audio_mcp.pipeline.planner import TASK_TO_STAGES, plan_auto
from audio_mcp.pipeline.runner import run_analysis
from audio_mcp.tools.schemas import (
    AnalyzeOptions, AnalyzeRequest, DetectEventsRequest, FeaturesRequest, InspectQualityRequest, TimelineRequest, TranscribeRequest,
)

log = logging.getLogger("audio_mcp.tools")



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
        segs = r["speaker_assign"]["segments"] if "speaker_assign" in r else r["transcription"]["segments"]
        a["transcription"] = {"text": r["transcription"]["text"], "segments": segs}
    if "diarization" in r:
        a["speakers"] = {k: r["diarization"][k] for k in ("speakers", "turns", "overlaps", "overlap_supported")}
    if "events" in r:
        a["sound_events"] = r["events"]["events"]
    if "features" in r:
        a["acoustic_features"] = r["features"]
    if "quality" in r:
        a["audio_quality"] = r["quality"]
    if "timeline" in r:
        a["timeline"] = r["timeline"]
    if "uncertainty" in r:
        a["uncertainty"] = r["uncertainty"]
    if not options.word_level_timestamps:
        a = _strip(a, {"words"})
    if not options.include_confidence:
        a = _strip(a, {"confidence", "confidence_kind"})
    return a


LATE_STAGES = ("timeline", "uncertainty", "summary")  # read whatever earlier stages produced


def order_stages(stages: list[str]) -> list[str]:
    return [s for s in stages if s not in LATE_STAGES] + [s for s in LATE_STAGES if s in stages]


def _respond(outcome, result, *, key="result", extra_warnings=()):
    return build_response(result=result, audio_id=outcome.record.audio_id, payload_key=key,
                          warnings=[*outcome.warnings, *extra_warnings], errors=outcome.errors,
                          processing=outcome.processing)


_RECOVERABLE = {"LOW_LEVEL", "HIGH_NOISE", "HUM", "DC_OFFSET"}


@tool_handler
def analyze_audio(app, req: AnalyzeRequest) -> dict:
    plan = None
    if req.tasks == "auto":
        record = app.store.ingest(req.source)
        plan = plan_auto(app, record, app.store.load(record.audio_id), goal=req.goal, budget=req.budget)
        tasks = [p["task"] for p in plan]
    else:
        tasks = req.tasks
    stages = ["metadata"]  # always report audio_metadata
    for t in tasks:
        stages += [s for s in TASK_TO_STAGES[t] if s not in stages]
    o = req.options
    outcome = run_analysis(app, req.source, order_stages(stages), {
        "language": o.language, "word_timestamps": o.word_level_timestamps, "num_speakers": o.num_speakers,
        "diarize": "speaker_diarization" in tasks, "budget": req.budget})
    warns = [warning("UNSUPPORTED_FEATURE", "emotion detection is not implemented")] if o.detect_emotions else []
    if plan is not None:
        outcome.processing.plan = plan
    payload = analysis_payload(outcome, o)
    q, t = outcome.results.get("quality"), outcome.results.get("transcription")
    if app.settings.auto_preprocess and q and t and t["segments"]:
        high = [i for i in q["issues"] if i["severity"] == "high" and i["code"] in _RECOVERABLE]
        ops = [op for op in q["recommendations"] if op != "mono"]
        if high and ops:
            audio = app.store.load(outcome.record.audio_id)
            payload["recovery"] = {"triggered_by": [i["code"] for i in high], "ops": ops,
                                   **compare_preprocessed_asr(app, outcome.record, audio, ops)}
    return _respond(outcome, payload, key="analysis", extra_warnings=warns)


@tool_handler
def transcribe_audio(app, req: TranscribeRequest) -> dict:
    stages = ["transcription"] + (["diarization", "speaker_assign"] if req.diarize else [])
    outcome = run_analysis(app, req.source, stages, {
        "language": req.language, "word_timestamps": req.word_timestamps, "multilingual": req.multilingual})
    warns = [warning("UNSUPPORTED_FEATURE", "translate_to_en is not implemented yet")] if req.translate_to_en else []
    r = outcome.results
    t = r.get("transcription")
    result = {k: t[k] for k in ("text", "segments", "language", "languages")} if t else None
    if result and "speaker_assign" in r:
        result["segments"] = r["speaker_assign"]["segments"]
    if result and "diarization" in r:
        result["speakers"] = {k: r["diarization"][k] for k in ("speakers", "turns", "overlaps", "overlap_supported")}
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


@tool_handler
def get_audio_timeline(app, req: TimelineRequest) -> dict:
    stages = (["diarization", "speaker_assign"] if req.diarize else []) + ["timeline"]
    outcome = run_analysis(app, req.source, stages, {"diarize": req.diarize, "timeline_include": req.include})
    return _respond(outcome, outcome.results.get("timeline"))

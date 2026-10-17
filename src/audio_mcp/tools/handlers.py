import functools
import logging
from pathlib import Path

from pydantic import ValidationError

import numpy as np

from audio_mcp.analysis.events import EVENT_CATEGORIES
from audio_mcp.compare.compare import baseline_anomalies, compare_pair, window_features
from audio_mcp.dsp.features import extract_features
from audio_mcp.dsp.preprocess import apply_preprocessing
from audio_mcp.ingest.decode import resample
from audio_mcp.envelope import AudioMCPError, ErrorCode, ProcessingInfo, build_response, error_response, warning
from audio_mcp.pipeline.planner import TASK_TO_STAGES, plan_auto
from audio_mcp.ingest.sources import FilePathSource, check_path_allowed
from audio_mcp.library.index import SUPPORTED_EXTENSIONS
from audio_mcp.pipeline.runner import run_analysis
from audio_mcp.reasoning.ask import answer_question, classify_intent
from audio_mcp.tools.schemas import (
    AnalyzeOptions, AnalyzeRequest, AskRequest, CompareRequest, DetectEventsRequest, FeaturesRequest, InspectQualityRequest, SearchLibraryRequest, SummarizeRequest, TimelineRequest, TranscribeRequest,
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
    if "summary" in r:
        a["summary"] = r["summary"]
    if "integrity" in r:
        a["integrity"] = r["integrity"]
    if "speech" in r:
        a["speech_characteristics"] = r["speech"]
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
    stages = ["transcription", "language"] + (["diarization", "speaker_assign"] if req.diarize else [])
    stages += ["translation"] if req.translate_to_en else []
    outcome = run_analysis(app, req.source, stages, {
        "language": req.language, "word_timestamps": req.word_timestamps, "multilingual": req.multilingual})
    warns = [warning("TRANSLATION_TARGET_LIMITED", "only English translation is supported locally")
             ] if req.translate_to_en else []
    r = outcome.results
    t = r.get("transcription")
    result = {k: t[k] for k in ("text", "segments", "language", "languages")} if t else None
    if result and "speaker_assign" in r:
        result["segments"] = r["speaker_assign"]["segments"]
    if result and "translation" in r:
        tr = {s["id"]: s.get("translation") for s in r["translation"]["segments"]}
        result["segments"] = [{**s, "translation": tr[s["id"]]} if tr.get(s["id"]) else s for s in result["segments"]]
    if result and "language" in r:
        result["code_switching"] = r["language"]["code_switching"]
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


def text_embedder(app, outcome):
    """The text embedder if usable; records it in models_used. Fake/real both qualify."""
    b = app.backends.text
    if not b.available()[0]:
        return None
    outcome.processing.models_used.append(b.info)
    return b


@tool_handler
def ask_audio(app, req: AskRequest) -> dict:
    if req.use_llm and not app.backends.llm.available()[0]:
        raise AudioMCPError(ErrorCode.LLM_DISABLED, "use_llm requested but no LLM is configured (llm_provider=none)")
    intent, _ = classify_intent(req.question)
    diarize = intent in ("speakers", "change")
    stages = (["diarization", "speaker_assign"] if diarize else []) + (["quality"] if intent == "unusual" else [])
    stages += ["timeline", "uncertainty"]
    outcome = run_analysis(app, req.source, stages, {"diarize": diarize})
    audio = app.store.load(outcome.record.audio_id)
    results = {**outcome.results, "metadata": {"duration_seconds": round(audio.duration, 3)}}
    answer = answer_question(req.question, results, audio.duration, text_embedder(app, outcome), audio=audio)
    if not req.include_evidence:
        answer.pop("evidence")
    return _respond(outcome, answer)


def _require_llm(app, use_llm):
    if use_llm and not app.backends.llm.available()[0]:
        raise AudioMCPError(ErrorCode.LLM_DISABLED, "use_llm requested but no LLM is configured (llm_provider=none)")


@tool_handler
def summarize_audio(app, req: SummarizeRequest) -> dict:
    _require_llm(app, req.use_llm)
    outcome = run_analysis(app, req.source, ["timeline", "summary"],
                           {"summary_length": req.length, "summary_focus": req.focus})
    return _respond(outcome, outcome.results.get("summary"))


def _merge_processing(outcomes) -> ProcessingInfo:
    p = ProcessingInfo()
    for o in outcomes:
        q = o.processing
        p.duration_seconds = round(p.duration_seconds + q.duration_seconds, 3)
        p.stages_run += [f"{o.record.audio_id}:{s}" for s in q.stages_run]
        p.cache_hits += [f"{o.record.audio_id}:{s}" for s in q.cache_hits]
        p.stages_skipped += [{**s, "audio_id": o.record.audio_id} for s in q.stages_skipped]
        p.models_used += [m for m in q.models_used if m not in p.models_used]
    return p


def _bundle(outcome, duration) -> dict:
    r = outcome.results
    return {"audio_id": outcome.record.audio_id, "duration": duration, "features": r.get("features") or {},
            "events": (r.get("events") or {}).get("events", []),
            "transcript": r["transcription"]["text"] if "transcription" in r else None,
            "embedding": (r.get("audio_embedding") or {}).get("embedding")}


@tool_handler
def compare_audio(app, req: CompareRequest) -> dict:
    aspects = set(req.aspects or ["features", "events", "embedding"])
    stages = ["metadata", "features", "events", "audio_embedding"] + (["transcription"] if "transcript" in aspects else [])
    outs = [run_analysis(app, s, stages, {}) for s in req.sources]
    warns = [{**w, "audio_id": o.record.audio_id} for o in outs for w in o.warnings]
    errors = [{**e, "audio_id": o.record.audio_id} for o in outs for e in o.errors]
    result = {"audio_ids": [o.record.audio_id for o in outs]}
    if req.baseline:
        base_outs = [run_analysis(app, s, ["metadata"], {}) for s in req.baseline]

        def wf(o):
            a = app.store.load(o.record.audio_id)
            return window_features(a.native, a.native_sr)

        base = [wf(o) for o in base_outs]
        result["baseline_ids"] = [o.record.audio_id for o in base_outs]
        result["baseline"] = {o.record.audio_id: baseline_anomalies(wf(o), base, z_threshold=req.z_threshold)
                              for o in outs}
        outs_all = outs + base_outs
    else:
        bundles = [_bundle(o, app.store.load(o.record.audio_id).duration) for o in outs]
        comps = []
        for b in bundles[1:]:
            c = compare_pair(bundles[0], b)
            c["measured_differences"] = {k: v for k, v in c["measured_differences"].items()
                                         if k in aspects or (k == "bands" and "features" in aspects)}
            if "embedding" in aspects and not c["similarity"]:
                warns.append(warning("EMBEDDING_UNAVAILABLE", "audio embeddings unavailable; similarity omitted"))
            comps.append(c)
        result["comparisons"] = comps
        outs_all = outs
    return build_response(result=result, warnings=warns, errors=errors, processing=_merge_processing(outs_all))


_INDEX_STAGES = ["transcription", "language", "events", "audio_embedding"]


def _expand(app, raw: str):
    """Yield (path, error) for a file or folder argument, enforcing allowed roots on every file."""
    try:
        yield check_path_allowed(raw, app.settings.allowed_roots), None
        return
    except AudioMCPError as e:
        if e.code != ErrorCode.INVALID_INPUT or not Path(raw).is_dir():  # INVALID_INPUT here = allowed directory
            yield raw, e
            return
    for f in sorted(Path(raw).resolve().rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                yield check_path_allowed(f, app.settings.allowed_roots), None
            except AudioMCPError as e:
                yield f, e


@tool_handler
def search_audio_library(app, req: SearchLibraryRequest) -> dict:
    lib = app.library
    if req.action == "list":
        return build_response(result={"recordings": lib.list_recordings(limit=req.limit)})
    if req.action == "remove":
        return build_response(result={"removed": lib.remove(req.audio_id)}, audio_id=req.audio_id)
    if req.action == "search":
        outs, emb = [], None
        if req.query_audio is not None:
            o = run_analysis(app, req.query_audio, ["audio_embedding"], {})
            outs.append(o)
            emb = (o.results.get("audio_embedding") or {}).get("embedding")
        filters = req.filters.model_dump(exclude_none=True) if req.filters else {}
        matches = lib.search(req.query, filters=filters, limit=req.limit, query_audio_embedding=emb)
        warns = [w for o in outs for w in o.warnings]
        if req.query_audio is not None and emb is None:
            warns.append(warning("EMBEDDING_UNAVAILABLE", "audio embedding unavailable; query_audio ignored"))
        return build_response(result={"matches": matches}, warnings=warns, processing=_merge_processing(outs))
    indexed, failed, outs = [], [], []
    for raw in req.paths:
        for path, err in _expand(app, raw):
            if err is None:
                try:
                    o = run_analysis(app, FilePathSource(type="file_path", path=str(path)), _INDEX_STAGES, {})
                    emb = (o.results.get("audio_embedding") or {}).get("embedding")
                    n = lib.add(o.record, Path(path), o.results, emb)
                    outs.append(o)
                    indexed.append({"path": str(path), "audio_id": o.record.audio_id, "segments": n})
                    continue
                except AudioMCPError as e:
                    err = e
            failed.append({"path": str(path), "error": err.to_dict()})
    return build_response(result={"indexed": indexed, "failed": failed}, processing=_merge_processing(outs))

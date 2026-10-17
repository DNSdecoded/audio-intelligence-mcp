import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from audio_mcp.context import AppContext
from audio_mcp.envelope import AudioMCPError, ErrorCode, ProcessingInfo, warning
from audio_mcp.ingest.sources import AudioSource
from audio_mcp.jobs import JOB_CONTEXT
from audio_mcp.pipeline.planner import topo_order
from audio_mcp.pipeline.stages import STAGES, StageContext
from audio_mcp.store.audio_store import AudioRecord
from audio_mcp.store.cache import config_hash


@dataclass
class AnalysisOutcome:
    record: AudioRecord
    results: dict[str, dict]
    processing: ProcessingInfo
    warnings: list[dict]
    errors: list[dict]


def run_analysis(app: AppContext, source: AudioSource, stages: list[str], options: dict, *,
                 cancel_event=None, progress=None) -> AnalysisOutcome:
    t0 = time.monotonic()
    job = JOB_CONTEXT.get()
    if job is not None:  # running inside an async job: inherit its cancel flag and progress reporting
        cancel_event = cancel_event or job[0]
        progress = progress or job[1]
    record = app.store.ingest(source)  # AudioMCPError propagates to the handler
    audio = app.store.load(record.audio_id)
    order = topo_order(stages)
    options = {**options, "_stages": order}  # lets late stages key their cache on what else ran
    ctx = StageContext(app, record, audio, options, cancel_event=cancel_event)
    ctx.warnings.extend(warning("DECODE_WARNING", w) for w in audio.decode_warnings)
    ran, skipped, hits, errors = [], [], [], []
    bad: dict[str, str] = {}  # stage -> "failed" | "skipped"
    for done, name in enumerate(order, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise AudioMCPError(ErrorCode.CANCELLED, "analysis cancelled")
        stage = STAGES.get(name)
        if stage is None:
            skipped.append({"stage": name, "reason": "not implemented in this version"})
            bad[name] = "skipped"
        else:
            _run_stage(app, stage, ctx, bad, ran, skipped, hits, errors)
        if progress:
            progress(done, len(order))
    results = {n: ctx.results[n] for n in order if n in ctx.results}
    info = ProcessingInfo(duration_seconds=round(time.monotonic() - t0, 3), stages_run=ran, stages_skipped=skipped,
                          cache_hits=hits, models_used=ctx.models_used)
    return AnalysisOutcome(record, results, info, ctx.warnings, errors)


def _run_stage(app, stage, ctx, bad, ran, skipped, hits, errors):
    key = config_hash(stage.cfg(app, ctx.options))
    cached = app.cache.get(ctx.record.audio_id, stage.name, key)
    if cached is not None:
        ctx.results[stage.name] = cached["result"]
        ctx.warnings.extend(cached["warnings"])
        hits.append(stage.name)
        return
    for d in stage.deps:
        if d in bad:
            skipped.append({"stage": stage.name, "reason": f"dependency {d} {bad[d]}"})
            bad[stage.name] = "skipped"
            return
    if stage.backend:
        ok, reason = getattr(app.backends, stage.backend).available()
        if not ok:
            skipped.append({"stage": stage.name, "reason": reason or "backend unavailable"})
            bad[stage.name] = "skipped"
            return
    n_warn = len(ctx.warnings)
    try:
        ctx.results[stage.name] = _call_with_timeout(stage.fn, ctx, app.settings.stage_timeout_seconds)
    except AudioMCPError as e:
        del ctx.warnings[n_warn:]
        if e.code != ErrorCode.MODEL_UNAVAILABLE:
            errors.append({**e.to_dict(), "stage": stage.name})
            bad[stage.name] = "failed"
        else:  # a model that can't load is a skip, not an error
            skipped.append({"stage": stage.name, "reason": e.message})
            bad[stage.name] = "skipped"
        return
    except Exception as e:  # one stage failing must not abort the others
        errors.append({"code": ErrorCode.STAGE_FAILED.value, "stage": stage.name, "message": str(e),
                       "retryable": False})
        bad[stage.name] = "failed"
        del ctx.warnings[n_warn:]
        return
    ran.append(stage.name)
    app.cache.put(ctx.record.audio_id, stage.name, key,
                  {"result": ctx.results[stage.name], "warnings": ctx.warnings[n_warn:]})


def _call_with_timeout(fn, ctx, timeout: float):
    # shortcut: a timed-out stage thread is abandoned, not killed (Python cannot kill threads); acceptable
    # because stages are pure computations on in-memory audio.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-mcp-stage")
    fut = pool.submit(fn, ctx)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        raise AudioMCPError(ErrorCode.TIMEOUT, f"stage exceeded {timeout:g} s", retryable=True) from None
    finally:
        pool.shutdown(wait=False)

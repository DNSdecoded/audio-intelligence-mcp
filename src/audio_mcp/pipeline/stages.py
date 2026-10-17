"""Stage registry: each stage is a function of the StageContext returning a JSON-able dict."""
import threading
from dataclasses import dataclass, field
from typing import Callable

from audio_mcp.analysis.events import frames_to_events
from audio_mcp.context import AppContext
from audio_mcp.dsp.features import FEATURE_GROUPS, extract_features
from audio_mcp.envelope import ModelInfo, warning
from audio_mcp.ingest.decode import DecodedAudio
from audio_mcp.store.audio_store import AudioRecord
from audio_mcp.types import Interval


@dataclass
class StageContext:
    app: AppContext
    record: AudioRecord
    audio: DecodedAudio
    options: dict
    results: dict[str, dict] = field(default_factory=dict)
    models_used: list[ModelInfo] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    cancel_event: threading.Event | None = None


@dataclass
class Stage:
    name: str
    deps: tuple[str, ...]
    fn: Callable[[StageContext], dict]
    cfg: Callable[[AppContext, dict], dict]
    backend: str | None = None  # Backends field that must be available
    soft_deps: tuple[str, ...] = ()  # scheduled first; their skip/failure does not skip this stage


def _regions(ctx) -> list[Interval]:
    return [Interval(**r) for r in ctx.results["vad"]["speech_regions"]]


def _id(b) -> str:
    return f"{b.info.name}@{b.info.version}"


def _metadata(ctx):
    return {**ctx.record.probe.to_dict(), "duration_seconds": round(ctx.audio.duration, 3)}


def _vad(ctx):
    b = ctx.app.backends.vad
    try:
        ok, reason = b.available()
        if not ok:
            raise RuntimeError(reason)
        regions = b.speech_regions(ctx.audio.mono16k)
    except Exception as e:  # any primary VAD failure falls back to the energy VAD
        ctx.warnings.append(warning("VAD_FALLBACK", f"primary VAD failed ({type(e).__name__}: {e}); used fallback VAD"))
        b = ctx.app.backends.vad_fallback
        regions = b.speech_regions(ctx.audio.mono16k)
    ctx.models_used.append(b.info)
    speech = sum(r.end - r.start for r in regions)
    return {"speech_regions": [r.model_dump(exclude_none=True) for r in regions],
            "speech_ratio": round(speech / max(ctx.audio.duration, 1e-9), 3), "backend": b.info.name}


def _features(ctx):
    groups = ctx.options.get("feature_groups") or FEATURE_GROUPS
    return extract_features(ctx.audio.native, ctx.audio.native_sr, groups, _regions(ctx))


def _transcription(ctx):
    regions = _regions(ctx)
    if not regions:
        ctx.warnings.append(warning("NO_SPEECH_DETECTED", "no speech detected; transcription skipped"))
        return {"text": "", "segments": [], "language": None, "language_probability": None, "languages": []}
    asr, o = ctx.app.backends.asr, ctx.options
    multi = bool(o.get("multilingual"))
    r = asr.transcribe(ctx.audio.mono16k, language=o.get("language"), word_timestamps=o.get("word_timestamps", True),
                       per_chunk_language=multi, chunks=regions if multi else None)
    ctx.models_used.append(asr.info)
    if r.dropped_low_speech:
        ctx.warnings.append(warning("LOW_SPEECH_DROPPED", f"dropped {r.dropped_low_speech} low-speech segments"))
    segs = [s.model_dump(exclude_none=True) for s in r.segments]
    return {"text": " ".join(s["text"].strip() for s in segs), "segments": segs, "language": r.language,
            "language_probability": r.language_probability, "languages": r.languages}


def _language(ctx):
    t = ctx.results["transcription"]
    out = {"primary": t["language"], "segments": t["languages"]}
    if t["language_probability"] is not None:
        out.update(confidence=t["language_probability"], confidence_kind="model_probability")
    return out


def _threshold(app, o):
    return o.get("event_threshold", app.settings.event_threshold)


def _events(ctx):
    o, tagger = ctx.options, ctx.app.backends.tagger
    thr = _threshold(ctx.app, o)
    tag = tagger.tag_windows(ctx.audio.mono16k)
    ctx.models_used.append(tagger.info)
    ev = frames_to_events(tag, duration=ctx.audio.duration, threshold=thr, labels=o.get("event_labels"))
    return {"events": [e.model_dump(exclude_none=True) for e in ev], "threshold": thr}


def _vad_cfg(app, o):
    return {"vad": _id(app.backends.vad), "fallback": _id(app.backends.vad_fallback)}


def _asr_cfg(app, o):  # also keys `language`, which only reads the transcription result
    return {**_vad_cfg(app, o), "asr": _id(app.backends.asr), "language": o.get("language"),
            "word_timestamps": o.get("word_timestamps", True), "multilingual": bool(o.get("multilingual"))}


STAGES: dict[str, Stage] = {s.name: s for s in [
    Stage("metadata", (), _metadata, lambda a, o: {}),
    Stage("vad", (), _vad, _vad_cfg),
    Stage("features", ("vad",), _features,
          lambda a, o: {**_vad_cfg(a, o), "groups": list(o.get("feature_groups") or FEATURE_GROUPS)}),
    Stage("transcription", ("vad",), _transcription, _asr_cfg, "asr"),
    Stage("language", ("transcription",), _language, _asr_cfg),
    Stage("events", (), _events,
          lambda a, o: {"tagger": _id(a.backends.tagger), "threshold": _threshold(a, o),
                        "labels": o.get("event_labels")}, "tagger"),
]}

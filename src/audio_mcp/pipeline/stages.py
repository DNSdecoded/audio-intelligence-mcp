"""Stage registry: each stage is a function of the StageContext returning a JSON-able dict."""
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from audio_mcp.analysis.diarize import assign_speakers, cluster_speakers, speaker_summary
from audio_mcp.analysis.events import frames_to_events
from audio_mcp.context import AppContext
from audio_mcp.dsp.features import FEATURE_GROUPS, extract_features, frame_rms
from audio_mcp.dsp.quality import diagnose_quality
from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo, warning
from audio_mcp.ingest.decode import DecodedAudio
from audio_mcp.store.audio_store import AudioRecord
from audio_mcp.timeline.graph import build_timeline
from audio_mcp.types import Interval, Segment


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


def _diarization(ctx):
    app, o = ctx.app, ctx.options
    pyannote = app.backends.diarizer if app.settings.diarization_backend == "pyannote" else None
    b = pyannote or app.backends.speaker
    ok, reason = b.available()
    if not ok:
        raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, reason or "diarization backend unavailable")
    if pyannote:
        turns, overlaps = pyannote.diarize(ctx.audio.mono16k, o.get("num_speakers"))
        overlaps = [i.model_dump(exclude_none=True) for i in overlaps]
    else:
        turns = cluster_speakers(ctx.audio.mono16k, _regions(ctx), b, threshold=app.settings.diarization_threshold,
                                 num_speakers=o.get("num_speakers"), max_speakers=app.settings.max_speakers)
        overlaps = None
        ctx.warnings.append(warning("OVERLAP_UNSUPPORTED", "overlapping speech is not detected by the ECAPA backend"))
    ctx.models_used.append(b.info)
    return {"turns": [t.model_dump(exclude_none=True) for t in turns], "speakers": speaker_summary(turns),
            "overlaps": overlaps, "overlap_supported": pyannote is not None, "backend": b.info.name}


def _speaker_assign(ctx):
    from audio_mcp.types import SpeakerTurn

    segs = [Segment(**s) for s in ctx.results["transcription"]["segments"]]
    turns = [SpeakerTurn(**t) for t in ctx.results["diarization"]["turns"]]
    return {"segments": [s.model_dump(exclude_none=True) for s in assign_speakers(segs, turns)]}


def _diar_cfg(app, o):
    b = app.backends.diarizer if app.settings.diarization_backend == "pyannote" else app.backends.speaker
    return {**_vad_cfg(app, o), "backend": _id(b) if b else None, "threshold": app.settings.diarization_threshold,
            "num_speakers": o.get("num_speakers"), "max_speakers": app.settings.max_speakers}


def _level_windows(audio) -> list[float | None]:
    rms = frame_rms(audio.mono16k, 16000, win=1.0, hop=1.0)
    return [round(20 * float(np.log10(r)), 1) if r > 0 else None for r in rms]


def _timeline(ctx):
    include = ctx.options.get("timeline_include")
    return build_timeline(ctx.results, ctx.audio.duration, include=set(include) if include else None,
                          level_db=_level_windows(ctx.audio))


def _quality(ctx):
    return diagnose_quality(ctx.audio.native, ctx.audio.native_sr, ctx.results["features"]["values"], _regions(ctx))


def _features_cfg(app, o):
    return {**_vad_cfg(app, o), "groups": list(o.get("feature_groups") or FEATURE_GROUPS)}


def _vad_cfg(app, o):
    return {"vad": _id(app.backends.vad), "fallback": _id(app.backends.vad_fallback)}


def _asr_cfg(app, o):  # also keys `language`, which only reads the transcription result
    return {**_vad_cfg(app, o), "asr": _id(app.backends.asr), "language": o.get("language"),
            "word_timestamps": o.get("word_timestamps", True), "multilingual": bool(o.get("multilingual"))}


STAGES: dict[str, Stage] = {s.name: s for s in [
    Stage("metadata", (), _metadata, lambda a, o: {}),
    Stage("vad", (), _vad, _vad_cfg),
    Stage("features", ("vad",), _features, _features_cfg),
    Stage("diarization", ("vad",), _diarization, _diar_cfg),
    Stage("speaker_assign", ("transcription", "diarization"), _speaker_assign,
          lambda a, o: {**_asr_cfg(a, o), **_diar_cfg(a, o)}),
    Stage("timeline", ("metadata", "features"), _timeline,
          lambda a, o: {**_asr_cfg(a, o), **_diar_cfg(a, o), "tagger": _id(a.backends.tagger),
                        "threshold": _threshold(a, o), "diarize": bool(o.get("diarize")),
                        "include": sorted(o.get("timeline_include") or [])},
          soft_deps=("vad", "transcription", "events")),
    Stage("quality", ("features", "vad"), _quality, lambda a, o: {**_features_cfg(a, o), "rules": 1}),
    Stage("transcription", ("vad",), _transcription, _asr_cfg, "asr"),
    Stage("language", ("transcription",), _language, _asr_cfg),
    Stage("events", (), _events,
          lambda a, o: {"tagger": _id(a.backends.tagger), "threshold": _threshold(a, o),
                        "labels": o.get("event_labels")}, "tagger"),
]}

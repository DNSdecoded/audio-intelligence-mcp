from audio_mcp.pipeline.stages import STAGES

TASK_TO_STAGES: dict[str, list[str]] = {
    "acoustic_features": ["metadata", "features"],
    "transcription": ["transcription"],
    "language_detection": ["language"],
    "sound_event_detection": ["events"],
    "audio_quality": ["quality"],
    "speaker_diarization": ["diarization", "speaker_assign"],
    "speech_characteristics": ["speech"],
    "integrity": ["integrity"],
    "timeline": ["timeline"],
    "summarization": ["summary"],
    "uncertainty": ["uncertainty"],
}


def topo_order(requested: list[str]) -> list[str]:
    """Requested stages plus (hard and soft) deps, dependencies first, requested order kept otherwise."""
    order: list[str] = []

    def visit(n: str):
        if n in order:
            return
        s = STAGES.get(n)
        for d in (*s.deps, *s.soft_deps) if s else ():
            visit(d)
        order.append(n)

    for n in requested:
        visit(n)
    return order


_DIAR_GOAL = ("meeting", "interview", "speaker", "conversation", "call")
_EVENT_GOAL = ("machine", "industrial", "environment", "alarm", "noise", "sound")
_SUMMARY_GOAL = ("summar", "key points", "meeting")
_INTEGRITY_GOAL = ("authentic", "edit", "tamper", "splice", "fake", "synthetic")


def _goal_has(goal: str, words) -> bool:
    return any(w in goal for w in words)


def plan_auto(app, record, audio, *, goal: str | None, budget: str) -> list[dict]:
    """Pick tasks from cheap signals (VAD speech ratio, duration) plus the goal text and budget; reasons included."""
    from audio_mcp.ingest.sources import AudioIdSource
    from audio_mcp.pipeline.runner import run_analysis  # runner imports this module

    # a no-op progress keeps this pre-run from reporting into the enclosing job (it would show 1/1 = 0.99)
    vad = run_analysis(app, AudioIdSource(type="audio_id", audio_id=record.audio_id), ["vad"], {},
                       progress=lambda done, total: None).results["vad"]
    ratio, dur, g = vad["speech_ratio"], audio.duration, (goal or "").lower()
    speech = ratio >= app.settings.speech_ratio_threshold
    plan = [{"task": "acoustic_features", "reason": "cheap; always run"},
            {"task": "audio_quality", "reason": "cheap; always run"}]

    def add(task, reason):
        plan.append({"task": task, "reason": reason})

    if speech:
        add("transcription", f"speech ratio {ratio:.2f} >= {app.settings.speech_ratio_threshold}")
        add("language_detection", "speech present")
        if budget != "fast" and (dur > 20 or _goal_has(g, _DIAR_GOAL)):
            add("speaker_diarization", "goal mentions speakers" if _goal_has(g, _DIAR_GOAL) else f"speech over {dur:.0f} s")
    if _goal_has(g, _EVENT_GOAL):
        add("sound_event_detection", "goal mentions sounds/environment")
    elif not (budget == "fast" and ratio > 0.8):
        add("sound_event_detection", "non-speech content possible" if ratio <= 0.8 else "balanced/thorough budget")
    if speech or _goal_has(g, _SUMMARY_GOAL):
        add("summarization", "speech present" if speech else "goal asks for a summary")
    if any(p["task"] in ("sound_event_detection", "speaker_diarization") for p in plan):
        add("timeline", "events or speaker turns to order in time")
    if _goal_has(g, _INTEGRITY_GOAL):
        add("integrity", "goal asks about editing/authenticity")
    if speech and budget != "fast":
        add("speech_characteristics", "speech present")
    if budget != "fast":
        add("uncertainty", "thorough cross-check" if budget == "thorough" else "report low-confidence and conflicts")
    return plan

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

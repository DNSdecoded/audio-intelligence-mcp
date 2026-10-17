"""Sound-event interval extraction from windowed tagger scores."""
import numpy as np

from audio_mcp.backends.base import TagResult
from audio_mcp.types import Event

EVENT_CATEGORIES: dict[str, list[str]] = {
    "alarm": ["Alarm", "Beep, bleep", "Siren", "Smoke detector, smoke alarm", "Buzzer", "Alarm clock", "Fire alarm", "Civil defense siren"],
    "impact": ["Bang", "Thump, thud", "Slam", "Smash, crash", "Explosion", "Gunshot, gunfire", "Knock", "Thunk"],
    "footsteps": ["Walk, footsteps", "Run", "Shuffle"],
    "door": ["Door", "Doorbell", "Slam", "Knock", "Sliding door", "Cupboard open or close", "Drawer open or close"],
    "animal": ["Dog", "Bark", "Cat", "Meow", "Bird", "Bird vocalization, bird call, bird song", "Roar", "Animal"],
    "machinery": ["Engine", "Motor", "Mechanical fan", "Tools", "Power tool", "Drill", "Sawing", "Printer"],
    "music": ["Music", "Musical instrument", "Singing", "Guitar", "Piano", "Drum", "Pop music"],
    "environment": ["Wind", "Rain", "Thunderstorm", "Water", "Stream", "Fire", "Ocean"],
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking", "Conversation", "Narration, monologue"],
    "vehicle": ["Vehicle", "Car", "Truck", "Bus", "Motorcycle", "Train", "Aircraft", "Siren"],
}


def frames_to_events(
    tag: TagResult,
    *,
    duration: float,
    threshold: float,
    labels: list[str] | None = None,
    exclude: frozenset[str] = frozenset(),
    source: str = "panns",
) -> list[Event]:
    scores = np.asarray(tag.scores)
    starts = np.asarray(tag.window_starts, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0 or not tag.labels:
        return []
    want = {x.lower() for x in labels} if labels else None
    skip = {x.lower() for x in exclude}
    raw = []  # (start, label, end, confidence)
    for j, label in enumerate(tag.labels):
        if (want is not None and label.lower() not in want) or label.lower() in skip:
            continue
        run: list[int] = []
        for i in [*np.flatnonzero(scores[:, j] >= threshold), -1]:  # -1 sentinel flushes the last run
            if run and (i < 0 or starts[i] > starts[run[-1]] + tag.window_seconds):
                raw.append((starts[run[0]], label, min(starts[run[-1]] + tag.window_seconds, duration),
                            float(scores[run, j].max())))
                run = []
            if i >= 0:
                run.append(int(i))
    raw.sort(key=lambda r: (r[0], r[1]))
    return [
        Event(id=f"evt_{n:04d}", label=label, start=round(float(s), 3), end=round(float(e), 3),
              confidence=round(c, 3), confidence_kind="model_probability", source=source)
        for n, (s, label, e, c) in enumerate(raw, 1)
    ]

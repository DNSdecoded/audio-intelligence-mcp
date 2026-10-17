import numpy as np

from audio_mcp.analysis.events import frames_to_events
from audio_mcp.backends.base import TagResult

TAG = TagResult(labels=["Speech", "Beep, bleep"], scores=np.array([[0.9, 0], [0.9, 0], [0, 0.8], [0, 0.1]]),
                window_starts=np.array([0, .5, 1, 1.5]), window_seconds=1.0, embeddings=np.zeros((4, 2)))


def test_merge_runs():
    ev = frames_to_events(TAG, duration=2.5, threshold=0.3)
    assert [(e.label, e.start, e.end, e.confidence) for e in ev] == [("Speech", 0.0, 1.5, 0.9), ("Beep, bleep", 1.0, 2.0, 0.8)]
    assert ev[0].id == "evt_0001" and ev[0].confidence_kind == "model_probability"


def test_label_filter_case_insensitive():
    assert [e.label for e in frames_to_events(TAG, duration=2.5, threshold=0.3, labels=["speech"])] == ["Speech"]


def test_exclude_clamp_and_empty():
    assert [e.label for e in frames_to_events(TAG, duration=2.5, threshold=0.3, exclude=frozenset({"Speech"}))] == ["Beep, bleep"]
    assert frames_to_events(TAG, duration=1.2, threshold=0.3, labels=["speech"])[0].end == 1.2
    empty = TagResult([], np.zeros((0, 0)), np.zeros(0), 1.0, np.zeros((0, 2)))
    assert frames_to_events(empty, duration=0.0, threshold=0.3) == []


def test_gap_splits_runs():
    t = TagResult(["A"], np.array([[.9], [0], [0], [.9]]), np.array([0, .5, 1, 1.5]), 1.0, np.zeros((4, 1)))
    assert [(e.start, e.end) for e in frames_to_events(t, duration=3, threshold=.3)] == [(0.0, 1.0), (1.5, 2.5)]

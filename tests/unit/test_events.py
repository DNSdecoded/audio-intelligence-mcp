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


def test_tag_windows_batches(settings):
    pytest.importorskip("torch")
    import math

    from audio_mcp.backends.panns import PannsTagger

    sizes = []

    class FakeModel:
        def inference(self, frames):
            sizes.append(len(frames))
            return np.zeros((len(frames), 527)), np.zeros((len(frames), 2048))

    t = PannsTagger(settings)
    t._model = FakeModel()
    r = t.tag_windows(np.zeros(10 * 16000, dtype=np.float32))
    n = math.ceil((320000 - 32000) / 16000) + 1
    assert max(sizes) <= 16 and sum(sizes) == n and len(r.scores) == n
    assert np.allclose(r.window_starts, np.arange(n) * 0.5)

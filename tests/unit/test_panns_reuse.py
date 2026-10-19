import numpy as np

from audio_mcp.backends.base import TagResult
from audio_mcp.backends.panns import PannsTagger


def test_embed_reuses_the_tagging_pass(settings, monkeypatch):
    """events and audio_embedding tag the same buffer: the CNN14 pass (_tag) must run once, not twice."""
    t, calls = PannsTagger(settings), []

    def fake_tag(a):  # stands in for the torch pass, so this runs without the models extra
        calls.append(a)
        n = 5
        return TagResult(labels=["x"] * 527, scores=np.zeros((n, 527)), window_starts=np.arange(n) * 0.5,
                         window_seconds=1.0, embeddings=np.ones((n, 2048)))

    monkeypatch.setattr(t, "_tag", fake_tag)
    x = np.zeros(16000 * 3, np.float32)
    t.tag_windows(x)
    assert t.embed(x).shape == (2048,) and len(calls) == 1  # reused for the same buffer
    t.tag_windows(np.ones(16000 * 3, np.float32))
    assert len(calls) == 2  # different audio is tagged afresh

import numpy as np

from audio_mcp.backends.panns import PannsTagger


class _Model:
    calls = 0

    def inference(self, frames):
        _Model.calls += 1
        return np.zeros((len(frames), 527), np.float32), np.ones((len(frames), 2048), np.float32)


def test_embed_reuses_the_tagging_pass(settings, monkeypatch):
    t = PannsTagger(settings)
    monkeypatch.setattr(t, "load", lambda: None)
    t._model = _Model()
    x = np.zeros(16000 * 3, np.float32)
    t.tag_windows(x)
    n = _Model.calls
    assert t.embed(x).shape == (2048,) and _Model.calls == n  # no second CNN14 pass for the same buffer
    t.tag_windows(np.ones(16000 * 3, np.float32))
    assert _Model.calls > n  # different audio is tagged afresh

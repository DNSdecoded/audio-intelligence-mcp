import pytest

from conftest import FIXTURES_DIR
from audio_mcp.eval.metrics import der, load_rttm


def test_der_perfect_and_swapped_labels():
    ref = [{"speaker_id": "A", "start": 0, "end": 5}, {"speaker_id": "B", "start": 5, "end": 10}]
    hyp = [{"speaker_id": "s2", "start": 0, "end": 5}, {"speaker_id": "s1", "start": 5, "end": 10}]
    assert der(ref, hyp)["der"] == pytest.approx(0, abs=1e-6)


def test_der_missed():
    assert der([{"speaker_id": "A", "start": 0, "end": 10}], [], collar=0)["der"] == pytest.approx(1.0)


def test_der_confusion_and_false_alarm():
    ref = [{"speaker_id": "A", "start": 0, "end": 10}]
    hyp = [{"speaker_id": "x", "start": 0, "end": 5}, {"speaker_id": "y", "start": 5, "end": 12}]
    r = der(ref, hyp, collar=0)
    assert r["confusion"] == pytest.approx(0.5, abs=0.01) and r["false_alarm"] == pytest.approx(0.2, abs=0.01)


def test_load_rttm():
    turns = load_rttm(FIXTURES_DIR / "dialogue_2spk.rttm")
    assert len(turns) == 4 and turns[1] == {"speaker_id": "zira", "start": 2.356, "end": pytest.approx(4.59)}

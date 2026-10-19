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


def test_event_prf():
    from audio_mcp.eval.metrics import event_prf

    ref = [{"label": "Dog", "start": 0, "end": 1}, {"label": "Car", "start": 2, "end": 3}]
    hyp = [{"label": "dog", "start": 0.1, "end": 1.0}, {"label": "Car", "start": 5, "end": 6}]
    assert event_prf(ref, hyp) == {"precision": .5, "recall": .5, "f1": .5, "tp": 1, "fp": 1, "fn": 1}
    assert event_prf([], []) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    alt = [{"label": "Dog|Bark", "start": 0, "end": 1}]
    assert event_prf(alt, [{"label": "bark", "start": 0, "end": 1}])["tp"] == 1


def test_error_rates_keep_combining_marks_and_count_chars():
    from audio_mcp.eval.metrics import _words, char_error_rate, word_error_rate

    assert _words("नमस्ते, दुनिया!") == ["नमस्ते", "दुनिया"]
    assert word_error_rate("नमस्ते दुनिया", "नमस्ते दुनिया") == 0.0
    assert char_error_rate("你好世界", "你好世间") == 0.25
    assert _words("It's done.") == ["it's", "done"]

from conftest import fp
from audio_mcp.export.redact import redact_analysis, redact_text
from audio_mcp.tools.handlers import transcribe_audio
from audio_mcp.tools.schemas import TranscribeRequest


def test_redact():
    t, n = redact_text("mail a.b@x.org or call +1 (555) 123-4567, card 4111 1111 1111 1111")
    assert t == "mail [EMAIL] or call [PHONE], card [CARD]" and n == 3


def test_non_luhn_number_is_not_a_card():
    t, n = redact_text("order 1234 5678 9012 3456")
    assert "[CARD]" not in t


def test_extra_patterns_and_nested():
    obj = {"segments": [{"text": "project falcon is late", "start": 1.0}], "label": "falcon", "answer": "falcon"}
    out, n = redact_analysis(obj, [r"falcon"])
    assert out["segments"][0]["text"] == "project [REDACTED] is late" and out["answer"] == "[REDACTED]"
    # labels are redacted too: timeline speech nodes carry the transcript in "label"
    assert out["label"] == "[REDACTED]" and n == 3 and obj["segments"][0]["text"] == "project falcon is late"


def test_redact_outputs_setting(app, speech_en):
    app.settings.redact_outputs = True
    app.settings.redact_extra_patterns = ["friday"]
    r = transcribe_audio(app, TranscribeRequest(source=fp(speech_en)))
    assert "friday" not in r["result"]["text"] and any(w["code"] == "REDACTED" for w in r["warnings"])


def test_redacts_timeline_labels_and_numbers_split_across_words():
    from audio_mcp.export.redact import redact_analysis

    words = [{"text": t, "start": i, "end": i + 1} for i, t in enumerate(["call", "me", "at", "555", "123", "4567", "thanks"])]
    obj = {"segments": [{"text": "call me at 555 123 4567 thanks", "words": words}],
           "timeline": {"nodes": [{"type": "speech_segment", "label": "call me at 555 123 4567 thanks"}]}}
    out, n = redact_analysis(obj)
    assert [w["text"] for w in out["segments"][0]["words"]] == ["call", "me", "at", "[PHONE]", "[PHONE]", "[PHONE]", "thanks"]
    assert "4567" not in str(out) and n == 3
    assert words[3]["text"] == "555"  # the input is not mutated

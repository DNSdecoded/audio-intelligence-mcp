from conftest import fp
from audio_mcp.reasoning.ask import classify_intent, resolve_labels
from audio_mcp.tools.handlers import ask_audio, extract_audio_features
from audio_mcp.tools.schemas import AskRequest


def test_classify():
    assert classify_intent("How many speakers are there?")[0] == "speakers"
    assert classify_intent("When does the speaker mention the deadline?") == ("time_of_mention", {"topic": "the deadline"})
    assert classify_intent("What happens immediately before the alarm sounds?")[1] == {"rel": "before", "target": "alarm"}
    assert classify_intent("Is there a dog barking?") == ("presence", {"target": "dog barking"})
    assert classify_intent("Are there any unusual sounds in the background?")[0] == "unusual"
    assert classify_intent("What changes between the beginning and the end?")[0] == "change"
    assert classify_intent("What colour is the sky?")[0] == "fallback"


def test_resolve_labels():
    assert {"Dog", "Bark"} <= set(resolve_labels("dog barking", ["Speech"]))
    assert "Beep, bleep" in resolve_labels("alarm", [])


def test_mention(app, speech_en):
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="When does the speaker mention the deadline Friday?"))["result"]
    ev = r["evidence"]
    assert r["support"] == "direct" and "friday" in ev[0]["text"].lower()
    assert ev[0]["start"] < ev[0]["end"] and ev[0]["source"] == "transcription"
    assert r["confidence_kind"] == "model_probability" and 0 <= r["confidence"] <= 1


def test_before_alarm(app, synth):
    r = ask_audio(app, AskRequest(source=fp(synth["alarm"]), question="What happens immediately before the alarm sounds?"))["result"]
    assert r["intent"] == "relative_to_event" and r["support"] == "direct"
    assert all(e["end"] <= 1.1 for e in r["evidence"] if e["source"] != "panns" and e["label"] != "Beep, bleep")


def test_absent_event(app, speech_en):
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="Is there a dog barking?"))["result"]
    assert r["answer"].startswith("No") and r["support"] == "direct"


def test_unanswerable(app, synth):
    r = ask_audio(app, AskRequest(source=fp(synth["noise"]), question="What is the speaker's favourite colour?"))["result"]
    assert r["support"] == "insufficient" and "confidence" not in r


def test_speakers_and_change(app, dialogue):
    r = ask_audio(app, AskRequest(source=fp(dialogue), question="How many speakers are there?"))["result"]
    assert r["intent"] == "speakers" and r["support"] == "direct" and "speaker" in r["answer"]
    c = ask_audio(app, AskRequest(source=fp(dialogue), question="What changes between the beginning and the end?"))["result"]
    assert c["intent"] == "change" and "dB" in c["answer"]


def test_unusual(app, synth):
    r = ask_audio(app, AskRequest(source=fp(synth["clipped"]), question="Is anything unusual in this recording?"))["result"]
    assert r["intent"] == "unusual" and any(e["kind"] == "interpretation" for e in r["evidence"])


def test_no_evidence_flag(app, speech_en):
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="When does the speaker mention the report?",
                                  include_evidence=False))
    assert "evidence" not in r["result"]


def test_llm_disabled(app, speech_en):
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="deadline?", use_llm=True))
    assert r["error"]["code"] == "LLM_DISABLED"


def test_unknown_audio_id(app):  # Review Focus #3
    r = ask_audio(app, AskRequest(source={"type": "audio_id", "audio_id": "aud_0000000000000000"}, question="what happens?"))
    assert r["error"]["code"] == "AUDIO_NOT_FOUND"


def test_audio_id_runs_on_demand(app, speech_en):  # Review Focus #3
    aid = extract_audio_features(app, FeaturesRequest_(speech_en))["audio_id"]
    r = ask_audio(app, AskRequest(source={"type": "audio_id", "audio_id": aid}, question="When does the speaker mention the report?"))
    assert r["status"] == "success" and "transcription" in r["processing"]["stages_run"]


def FeaturesRequest_(p):
    from audio_mcp.tools.schemas import FeaturesRequest

    return FeaturesRequest(source=fp(p))

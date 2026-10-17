from conftest import fp
from audio_mcp.timeline.graph import build_timeline, render_text
from audio_mcp.tools.handlers import analyze_audio, extract_audio_features, get_audio_timeline
from audio_mcp.tools.schemas import AnalyzeRequest, FeaturesRequest, TimelineRequest


def seg(i, s, e, text, spk=None):
    d = {"id": i, "start": s, "end": e, "text": text}
    return {**d, "speaker_id": spk} if spk else d


def evt(i, label, s, e, c):
    return {"id": i, "label": label, "start": s, "end": e, "confidence": c, "confidence_kind": "model_probability",
            "source": "panns"}


RESULTS = {"transcription": {"segments": [seg("seg_0001", 0, 5, "hello"), seg("seg_0002", 9, 12, "again")]},
           "events": {"events": [evt("evt_0001", "Door", 5, 7, .8), evt("evt_0002", "Alarm", 7, 9, .9),
                                 evt("evt_0003", "Cough", 3, 3.5, .6), evt("evt_0004", "Speech", 0, 5, .9)]},
           "features": {"values": {"silence_intervals": []}}}


def test_sequence_and_relations():
    tl = build_timeline(RESULTS, 12.0)
    ids = [n["id"] for n in tl["nodes"] if n["type"] in {"speech_segment", "sound_event"}]
    assert ids == ["seg_0001", "evt_0003", "evt_0001", "evt_0002", "seg_0002"]  # Speech event dropped
    rel = {(e["source"], e["target"], e["relation"]) for e in tl["edges"]}
    assert ("evt_0001", "evt_0002", "precedes") in rel and ("evt_0003", "seg_0001", "interrupts") in rel
    assert "00:05.0–00:07.0" in tl["text"]
    gap = next(e for e in tl["edges"] if e["source"] == "evt_0002" and e["relation"] == "precedes")
    assert gap["target"] == "seg_0002" and gap["gap_seconds"] == 0.0


def test_turns_transitions_silence_and_include():
    r = {**RESULTS, "diarization": {"turns": [{"speaker_id": "speaker_1", "start": 0, "end": 5},
                                              {"speaker_id": "speaker_2", "start": 9, "end": 12}], "overlaps": None},
         "features": {"values": {"silence_intervals": [{"start": 7.5, "end": 8.5}]}}}
    tl = build_timeline(r, 12.0, level_db=[-20.0] * 6 + [-40.0] * 6)
    types = {n["type"] for n in tl["nodes"]}
    assert {"speaker_turn", "silence", "transition"} <= types
    labels = [n["label"] for n in tl["nodes"] if n["type"] == "transition"]
    assert any(l.startswith("speaker_change") for l in labels) and any(l.startswith("level_jump") for l in labels)
    assert any(l == "onset: Alarm" for l in labels)
    only = build_timeline(r, 12.0, include={"silence"})
    assert {n["type"] for n in only["nodes"]} == {"silence"}
    assert render_text(only).count("\n") == 0


def test_audio_id_on_demand(app, speech_en):  # Review Focus #3
    aid = extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["audio_id"]
    r = get_audio_timeline(app, TimelineRequest(source={"type": "audio_id", "audio_id": aid}))
    assert r["status"] == "success" and any(n["type"] == "speech_segment" for n in r["result"]["nodes"])
    assert any(n["type"] == "speaker_turn" for n in r["result"]["nodes"])


def test_timeline_survives_missing_asr(app, speech_en):
    from audio_mcp.backends.base import UnavailableBackend

    app.backends.asr = UnavailableBackend("faster-whisper", "not installed")
    r = get_audio_timeline(app, TimelineRequest(source=fp(speech_en), diarize=False))
    assert r["status"] == "success" and r["result"]["nodes"]
    assert any(s["stage"] == "transcription" for s in r["processing"]["stages_skipped"])


def test_analyze_timeline(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["timeline", "speaker_diarization"]))
    assert any(n["type"] == "speaker_turn" for n in r["analysis"]["timeline"]["nodes"])

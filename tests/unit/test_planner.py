from conftest import fp
from audio_mcp.ingest.sources import FilePathSource
from audio_mcp.pipeline.planner import plan_auto
from audio_mcp.tools.handlers import analyze_audio
from audio_mcp.tools.schemas import AnalyzeRequest


def plan_tasks(app, path, goal, budget):
    rec = app.store.ingest(FilePathSource(type="file_path", path=str(path)))
    return {p["task"] for p in plan_auto(app, rec, app.store.load(rec.audio_id), goal=goal, budget=budget)}


def test_speech_plan(app, speech_en):
    t = plan_tasks(app, speech_en, "meeting notes", "balanced")
    assert {"transcription", "speaker_diarization", "summarization", "uncertainty"} <= t


def test_noise_plan(app, synth):
    t = plan_tasks(app, synth["impulses"], "machine noise", "balanced")
    assert "sound_event_detection" in t and "transcription" not in t


def test_fast_budget(app, speech_en):
    t = plan_tasks(app, speech_en, None, "fast")
    assert "speaker_diarization" not in t and "uncertainty" not in t and "speech_characteristics" not in t


def test_integrity_goal(app, synth):
    assert "integrity" in plan_tasks(app, synth["spliced"], "was this edited or spliced?", "balanced")


def test_plan_reported(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks="auto"))
    assert r["processing"]["plan"] and all({"task", "reason"} <= p.keys() for p in r["processing"]["plan"])
    assert "transcription" in r["analysis"]


def test_recovery_when_enabled(app, speech_en, tmp_path):
    import soundfile as sf

    x, sr = sf.read(speech_en, dtype="float32")
    p = tmp_path / "dc.wav"
    sf.write(p, x + 0.03, sr)  # strong DC offset: a high-severity, recoverable issue
    req = AnalyzeRequest(source=fp(p), tasks=["transcription", "audio_quality"])
    assert "recovery" not in analyze_audio(app, req)["analysis"]
    app.settings.auto_preprocess = True
    rec = analyze_audio(app, req)["analysis"]["recovery"]
    assert rec["triggered_by"] == ["DC_OFFSET"] and "remove_dc" in rec["ops"]
    assert {"before", "after", "note"} <= rec.keys()

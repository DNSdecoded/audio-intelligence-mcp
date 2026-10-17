import numpy as np

from conftest import fp, load
from audio_mcp.dsp.integrity import INTEGRITY_LIMITATIONS, integrity_indicators
from audio_mcp.tools.handlers import analyze_audio
from audio_mcp.tools.schemas import AnalyzeRequest


def test_splice(synth):
    o = integrity_indicators(*load(synth["spliced"]))
    types = {(i["type"], round(i["start"], 1)) for i in o["observations"]}
    assert ("ZERO_RUN", 1.0) in types and any(t == "NOISE_FLOOR_JUMP" and abs(s - 2.0) <= 1.0 for t, s in types)
    assert "not evidence" in o["limitations"] and o["synthetic_speech_detection"]["status"] == "not_available"
    assert all({"type", "start", "end", "magnitude", "unit", "description"} <= i.keys() for i in o["observations"])


def test_clean_sine_quiet(synth):
    assert integrity_indicators(*load(synth["sine"]))["observations"] == []


def test_silence_quiet(synth):
    assert integrity_indicators(*load(synth["silence"]))["observations"] == []


def test_dc_step():
    x = np.full(16000 * 3, 0.1, dtype="float32")
    x[16000 * 2:] += 0.05
    obs = integrity_indicators(x[None], 16000)["observations"]
    assert any(o["type"] == "DC_STEP" and o["start"] == 2.0 for o in obs)


def test_limitations_text_exact():
    assert INTEGRITY_LIMITATIONS.startswith("These are signal-level observations")


def test_analyze_maps_integrity(app, synth):
    r = analyze_audio(app, AnalyzeRequest(source=fp(synth["spliced"]), tasks=["integrity"]))
    assert r["analysis"]["integrity"]["observations"]

import csv
import hashlib
import json
from pathlib import Path

import pytest

from conftest import fp
from audio_mcp.envelope import AudioMCPError
from audio_mcp.export.exporters import export_analysis
from audio_mcp.tools.handlers import export_audio_analysis, extract_audio_features
from audio_mcp.tools.schemas import ExportRequest, FeaturesRequest

AID = "aud_0123456789abcdef"
A = {"audio_metadata": {"duration_seconds": 5.0, "sample_rate_hz": 16000, "channels": 1},
     "transcription": {"text": "We need to finish.", "segments": [
         {"id": "seg_0001", "start": 1.2, "end": 4.8, "text": "We need to finish.", "speaker_id": "speaker_1",
          "confidence": 0.9, "confidence_kind": "model_probability"}]},
     "sound_events": [{"id": "evt_0001", "label": "Door", "start": 0.5, "end": 0.9, "confidence": 0.7,
                       "confidence_kind": "model_probability", "source": "panns"}],
     "summary": {"text": "We need to finish."}}


def ex(fmt, tmp_path, a=A, **kw):
    return export_analysis(AID, a, fmt, tmp_path, redact=kw.get("redact", False), extra_patterns=kw.get("extra", []))


def test_srt_vtt(tmp_path):
    s = Path(ex("srt", tmp_path)["path"]).read_text(encoding="utf-8")
    assert s.startswith("1\n") and "00:00:01,200 --> 00:00:04,800" in s and "[speaker_1] We need to finish." in s
    v = Path(ex("vtt", tmp_path)["path"]).read_text(encoding="utf-8")
    assert v.startswith("WEBVTT") and "00:00:01.200 --> 00:00:04.800" in v and "<v speaker_1>" in v


def test_rttm_line(tmp_path):
    a = {"speakers": {"turns": [{"speaker_id": "speaker_1", "start": 0.5, "end": 2.0}]}}
    t = Path(ex("rttm", tmp_path, a)["path"]).read_text(encoding="utf-8")
    assert t.strip() == f"SPEAKER {AID} 1 0.500 1.500 <NA> <NA> speaker_1 <NA> <NA>"
    with pytest.raises(AudioMCPError) as e:
        ex("rttm", tmp_path, {})
    assert e.value.code == "INVALID_INPUT"


def test_csv_and_markdown_and_json(tmp_path):
    with Path(ex("csv", tmp_path)["path"]).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["type"] for r in rows] == ["event", "speech"] and rows[1]["speaker_id"] == "speaker_1"
    assert list(rows[0]) == ["type", "id", "start", "end", "speaker_id", "label", "text", "confidence"]
    md = Path(ex("markdown", tmp_path)["path"])
    assert md.suffix == ".md" and "**[0:01.2] speaker_1:** We need to finish." in md.read_text(encoding="utf-8")
    j = ex("json", tmp_path)
    assert json.loads(Path(j["path"]).read_text(encoding="utf-8")) == A and j["redactions"] == 0


def test_redacted_export(tmp_path):
    out = ex("json", tmp_path, redact=True, extra=["finish"])
    assert "finish" not in Path(out["path"]).read_text(encoding="utf-8") and out["redactions"] >= 1


def test_export_tool_hash(app, speech_en):
    aid = extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["audio_id"]
    r = export_audio_analysis(app, ExportRequest(audio_id=aid, format="json"))
    res = r["result"]
    assert r["status"] == "success" and r["audio_id"] == aid
    assert hashlib.sha256(Path(res["path"]).read_bytes()).hexdigest() == res["sha256"]
    assert Path(res["path"]).parent == app.settings.resolved_export_dir


def test_export_srt_tool(app, dialogue):
    aid = extract_audio_features(app, FeaturesRequest(source=fp(dialogue)))["audio_id"]
    res = export_audio_analysis(app, ExportRequest(audio_id=aid, format="srt"))["result"]
    assert "[speaker_1]" in Path(res["path"]).read_text(encoding="utf-8")


def test_unknown_audio_id(app):
    r = export_audio_analysis(app, ExportRequest(audio_id="aud_0000000000000000", format="json"))
    assert r["error"]["code"] == "AUDIO_NOT_FOUND"

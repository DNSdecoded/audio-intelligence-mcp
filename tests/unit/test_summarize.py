from conftest import fp
from audio_mcp.reasoning.summarize import summarize
from audio_mcp.tools.handlers import analyze_audio, summarize_audio, transcribe_audio
from audio_mcp.tools.schemas import AnalyzeRequest, SummarizeRequest, TranscribeRequest


def test_extractive_grounded(app, dialogue):
    r = summarize_audio(app, SummarizeRequest(source=fp(dialogue), length="short"))
    seg_ids = {s["id"] for s in transcribe_audio(app, TranscribeRequest(source=fp(dialogue)))["result"]["segments"]}
    res = r["result"]
    assert len(res["sentences"]) == 2 and res["method"] == "extractive"
    assert all(set(s["evidence_ids"]) <= seg_ids for s in res["sentences"])
    assert res["topics"] and all(t["evidence_ids"] for t in res["topics"])
    assert res["key_moments"][0]["description"].startswith("Speech begins")


def test_time_order_and_length():
    segs = [{"id": f"seg_{i:04d}", "start": i, "end": i + 1, "text": t} for i, t in enumerate(
        ["budget review today", "the budget numbers are late", "lunch at noon", "budget approval needed", "ok"])]
    out = summarize({"transcription": {"segments": segs}}, 5.0, None, length="medium")
    starts = [int(s["evidence_ids"][0][-1]) for s in out["sentences"]]
    assert len(out["sentences"]) == 4 and starts == sorted(starts)
    focused = summarize({"transcription": {"segments": segs}}, 5.0, None, length="short", focus="lunch")
    assert any("lunch" in s["text"] for s in focused["sentences"])


def test_nonspeech_summary(app, synth):
    r = summarize_audio(app, SummarizeRequest(source=fp(synth["alarm"])))["result"]
    assert "Beep" in r["text"] and r["key_moments"] and r["sentences"][0]["evidence_ids"]


def test_llm_disabled(app, speech_en):
    assert summarize_audio(app, SummarizeRequest(source=fp(speech_en), use_llm=True))["error"]["code"] == "LLM_DISABLED"


def test_analyze_summary_and_speech(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["summarization", "speech_characteristics"]))
    a = r["analysis"]
    assert a["summary"]["sentences"] and a["speech_characteristics"]["words_per_minute"] > 0

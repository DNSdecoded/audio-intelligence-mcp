import pytest

from conftest import fp
from audio_mcp.backends.llm import AnthropicLLM, NoLLM, OpenAICompatibleLLM, is_loopback
from audio_mcp.envelope import ModelInfo
from audio_mcp.reasoning.llm_grounded import llm_answer, llm_summarize
from audio_mcp.tools.handlers import ask_audio, compare_audio, summarize_audio
from audio_mcp.tools.schemas import AskRequest, CompareRequest, SummarizeRequest
from audio_mcp.types import Evidence


def test_loopback():
    assert is_loopback("http://localhost:11434/v1") and is_loopback("http://127.0.0.1:8000")
    assert not is_loopback("https://api.anthropic.com")


class StubLLM:
    is_remote = True
    info = ModelInfo(stage="llm", name="stub", version="1", local=False)

    def __init__(self, out):
        self.out, self.prompts = out, []

    def available(self):
        return True, None

    def load(self):
        pass

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        return self.out


PACK = [Evidence(id="seg_0001", start=1, end=2, source="transcription", text="report by friday")]


def test_drops_uncited_claims():
    out = llm_answer("deadline?", PACK, StubLLM(
        '{"answer":"Friday","claims":[{"text":"Friday","evidence_ids":["seg_0001"],"kind":"detected"},'
        '{"text":"Boss angry","evidence_ids":["seg_9"],"kind":"interpretation"}],"insufficient":false}'))
    assert [c["text"] for c in out["claims"]] == ["Friday"] and out["support"] == "direct" and out["method"] == "llm"
    assert [e["id"] for e in out["evidence"]] == ["seg_0001"]


def test_unparseable_raises_value_error():
    with pytest.raises(ValueError):
        llm_answer("deadline?", PACK, StubLLM("not json"))


def test_summarize_keeps_cited_sentences():
    out = llm_summarize({"transcription": {"segments": []}}, PACK, StubLLM(
        '{"sentences":[{"text":"Report due Friday.","evidence_ids":["seg_0001"]},{"text":"x","evidence_ids":[]}],'
        '"topics":["deadline"]}'), "short", None)
    assert [s["text"] for s in out["sentences"]] == ["Report due Friday."] and out["method"] == "llm"


def test_remote_flag_and_disabled(app, speech_en):
    assert ask_audio(app, AskRequest(source=fp(speech_en), question="deadline?", use_llm=True))["error"]["code"] == "LLM_DISABLED"
    app.backends.llm = StubLLM('{"answer":"x","claims":[],"insufficient":true}')
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="deadline?", use_llm=True))
    assert r["processing"]["data_sent_off_machine"] is True and r["result"]["support"] == "insufficient"
    assert any(m["name"] == "stub" for m in r["processing"]["models_used"])


def test_local_llm_not_off_machine(app, speech_en):
    llm = StubLLM('{"answer":"x","claims":[],"insufficient":true}')
    llm.is_remote = False
    app.backends.llm = llm
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="deadline?"))
    assert r["processing"]["data_sent_off_machine"] is False and r["result"]["method"] == "llm"


def test_invalid_output_falls_back(app, speech_en):
    app.backends.llm = StubLLM("not json")
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="When does the speaker mention the report?", use_llm=True))
    assert any(w["code"] == "LLM_OUTPUT_INVALID" for w in r["warnings"]) and r["result"]["support"] == "direct"


def test_use_llm_false_skips_llm(app, speech_en):
    llm = StubLLM("{}")
    app.backends.llm = llm
    ask_audio(app, AskRequest(source=fp(speech_en), question="deadline?", use_llm=False))
    assert llm.prompts == []


def test_summarize_and_compare_with_llm(app, speech_en, synth):
    app.backends.llm = StubLLM('{"sentences":[{"text":"S.","evidence_ids":["seg_0001"]}],"topics":[],'
                               '"interpretations":[{"statement":"B is noisier.","based_on":["rms_dbfs"]}]}')
    s = summarize_audio(app, SummarizeRequest(source=fp(speech_en)))
    assert s["result"]["method"] == "llm" and s["processing"]["data_sent_off_machine"] is True
    c = compare_audio(app, CompareRequest(sources=[fp(synth["sine"]), fp(synth["noise"])]))
    interp = c["result"]["comparisons"][0]["possible_interpretations"]
    assert any(i["source"] == "llm" and i["kind"] == "interpretation" for i in interp)


def test_backends_selected(settings, monkeypatch):
    from audio_mcp.backends.base import build_backends

    settings.backend_profile = "real"
    assert isinstance(build_backends(settings).llm, NoLLM)
    settings.llm_provider, settings.llm_model = "ollama", "llama3"
    settings.llm_base_url = "http://localhost:11434/v1"
    llm = build_backends(settings).llm
    assert isinstance(llm, OpenAICompatibleLLM) and llm.is_remote is False and llm.info.local is True
    settings.llm_provider, settings.llm_model, settings.llm_base_url = "anthropic", "claude-opus-5-5", None
    llm = build_backends(settings).llm
    assert isinstance(llm, AnthropicLLM) and llm.is_remote is True


@pytest.mark.parametrize("out", [
    '{"sentences":[{"sentence":"no text key","evidence_ids":["seg_0001"]}],"topics":7}',
    '{"sentences":[{"text":"nested ids","evidence_ids":[["seg_0001"]]}]}',
    '{"sentences":"not a list","topics":[{"a":1}]}',
])
def test_summarize_wrong_shapes_are_dropped_not_raised(out):
    res = llm_summarize({"transcription": {"segments": []}}, PACK, StubLLM(out), "short", None)
    assert res["sentences"] == [] and res["topics"] == []


@pytest.mark.parametrize("out", ['{"answer":"x","claims":5}', '{"answer":"x","claims":[{"evidence_ids":"seg_0001"}]}'])
def test_answer_wrong_shapes_are_insufficient(out):
    assert llm_answer("q?", PACK, StubLLM(out))["support"] == "insufficient"


def test_interpret_wrong_shapes():
    from audio_mcp.reasoning.llm_grounded import llm_interpret

    assert llm_interpret({}, StubLLM('{"interpretations":{"statement":"x"}}')) == []
    assert llm_interpret({}, StubLLM('{"interpretations":[{"statement":"maybe","based_on":7}]}'))[0]["based_on"] == []

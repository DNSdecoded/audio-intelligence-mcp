import numpy as np
import pytest

from conftest import fp
from audio_mcp.analysis.diarize import assign_speakers, cluster_speakers
from audio_mcp.tools.handlers import analyze_audio, transcribe_audio
from audio_mcp.tools.schemas import AnalyzeRequest, TranscribeRequest
from audio_mcp.types import Interval, Segment, SpeakerTurn, Word


class SignEmbedder:
    def embed(self, ws):
        return np.array([[1, 0] if w.mean() > 0 else [0, 1] for w in ws], dtype="float32")


def test_two_clusters_synthetic():
    x = np.concatenate([np.full(48000, .1), np.full(48000, -.1), np.full(48000, .1)]).astype("float32")
    turns = cluster_speakers(x, [Interval(start=0, end=9)], SignEmbedder(), threshold=0.5, num_speakers=None, max_speakers=8)
    assert [t.speaker_id for t in turns] == ["speaker_1", "speaker_2", "speaker_1"]
    assert turns[1].start == pytest.approx(3.0, abs=0.8)


def test_num_speakers_and_cap():
    x = np.concatenate([np.full(48000, .1), np.full(48000, -.1)]).astype("float32")
    one = cluster_speakers(x, [Interval(start=0, end=6)], SignEmbedder(), threshold=0.5, num_speakers=1, max_speakers=8)
    assert {t.speaker_id for t in one} == {"speaker_1"}
    capped = cluster_speakers(x, [Interval(start=0, end=6)], SignEmbedder(), threshold=0.5, num_speakers=None, max_speakers=1)
    assert {t.speaker_id for t in capped} == {"speaker_1"}


def test_short_and_empty():
    x = np.full(8000, .1, dtype="float32")
    t = cluster_speakers(x, [Interval(start=0, end=0.5)], SignEmbedder(), threshold=0.5, num_speakers=None, max_speakers=8)
    assert [(s.speaker_id, s.start, s.end) for s in t] == [("speaker_1", 0.0, 0.5)]
    assert cluster_speakers(x, [], SignEmbedder(), threshold=0.5, num_speakers=None, max_speakers=8) == []


def test_assign_by_overlap():
    segs = [Segment(id="seg_0001", start=0, end=2, text="a",
                    words=[Word(start=0, end=1, text="a"), Word(start=1.9, end=2.0, text="b")]),
            Segment(id="seg_0002", start=2.5, end=4, text="b"), Segment(id="seg_0003", start=8, end=9, text="c")]
    out = assign_speakers(segs, [SpeakerTurn(speaker_id="speaker_1", start=0, end=2.2),
                                 SpeakerTurn(speaker_id="speaker_2", start=2.2, end=5)])
    assert [s.speaker_id for s in out] == ["speaker_1", "speaker_2", None]
    assert [w.speaker_id for w in out[0].words] == ["speaker_1", "speaker_1"]


def test_transcribe_diarize(app, dialogue):
    r = transcribe_audio(app, TranscribeRequest(source=fp(dialogue), diarize=True))
    res = r["result"]
    assert r["status"] == "success" and res["speakers"]["speakers"]
    assert all("speaker_id" in s for s in res["segments"])
    assert any(w["code"] == "OVERLAP_UNSUPPORTED" for w in r["warnings"])


def test_analyze_speakers(app, dialogue):
    r = analyze_audio(app, AnalyzeRequest(source=fp(dialogue), tasks=["transcription", "speaker_diarization"]))
    sp = r["analysis"]["speakers"]
    assert sp["overlap_supported"] is False and sp["turns"][0]["speaker_id"] == "speaker_1"
    assert {"speaker_id", "first_seen", "last_seen", "total_speech_seconds"} <= sp["speakers"][0].keys()


def test_short_window_joins_nearest_cluster():
    # 2 s of +, 0.5 s gap, 0.4 s of - (short), 0.5 s gap, 2 s of -  -> short window joins the "-" cluster
    x = np.zeros(int(16000 * 5.4), dtype="float32")
    x[:32000], x[40000:46400], x[54400:86400] = .1, -.1, -.1
    regions = [Interval(start=0, end=2), Interval(start=2.5, end=2.9), Interval(start=3.4, end=5.4)]
    turns = cluster_speakers(x, regions, SignEmbedder(), threshold=0.5, num_speakers=None, max_speakers=8)
    assert [t.speaker_id for t in turns] == ["speaker_1", "speaker_2", "speaker_2"]

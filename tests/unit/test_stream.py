import base64

import numpy as np
import soundfile as sf

from audio_mcp.tools.handlers import analyze_audio_stream
from audio_mcp.tools.schemas import StreamRequest


def stream(app, **kw):
    return analyze_audio_stream(app, StreamRequest(**kw))


def test_pcm_session(app, speech_en):
    x, sr = sf.read(speech_en, dtype="int16")
    sid = stream(app, action="start", chunk_format="pcm", sample_rate_hz=sr, channels=1,
                 sample_format="s16le")["result"]["session_id"]
    for part in np.array_split(x, 4):
        r = stream(app, action="push", session_id=sid, data=base64.b64encode(part.tobytes()).decode())["result"]
        assert r["provisional"] is True and r["chunk_end"] > r["chunk_start"]
    f = stream(app, action="finish", session_id=sid, tasks=["transcription", "sound_event_detection"])
    assert f["status"] == "success" and "reconciliation" in f["result"] and f["audio_id"].startswith("aud_")
    assert abs(f["result"]["analysis"]["audio_metadata"]["duration_seconds"] - len(x) / sr) < 0.05
    rec = f["result"]["reconciliation"]
    assert 0 <= rec["transcript_word_difference_rate"] and {"events_confirmed", "events_added", "events_dropped"} <= rec.keys()
    assert stream(app, action="status", session_id=sid)["error"]["code"] == "INVALID_INPUT"


def test_file_chunks(app, synth):
    sid = stream(app, action="start", chunk_format="file")["result"]["session_id"]
    for k in ["sine", "alarm"]:
        stream(app, action="push", session_id=sid, path=str(synth[k]))
    st = stream(app, action="status", session_id=sid)["result"]
    assert st["chunks"] == 2 and abs(st["duration_seconds"] - 6.0) < 0.05
    assert stream(app, action="cancel", session_id=sid)["result"]["state"] == "cancelled"


def test_pcm_requires_params(app):
    assert stream(app, action="start", chunk_format="pcm")["error"]["code"] == "INVALID_INPUT"


def test_push_needs_exactly_one_payload(app):
    sid = stream(app, action="start", chunk_format="file")["result"]["session_id"]
    assert stream(app, action="push", session_id=sid)["error"]["code"] == "INVALID_INPUT"


def test_path_outside_roots_rejected(app, tmp_path):
    sid = stream(app, action="start", chunk_format="file")["result"]["session_id"]
    r = stream(app, action="push", session_id=sid, path=str(tmp_path.parent / "x.wav"))
    assert r["error"]["code"] in {"PATH_NOT_ALLOWED", "FILE_NOT_FOUND"}


def test_expire(app):
    sid = stream(app, action="start", chunk_format="file")["result"]["session_id"]
    assert app.streams.expire(idle_seconds=-1) == 1
    assert stream(app, action="status", session_id=sid)["error"]["code"] == "INVALID_INPUT"

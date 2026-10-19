import base64

import numpy as np
import pytest
import soundfile as sf

from audio_mcp.envelope import AudioMCPError

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


def test_concurrent_pushes_and_close(app):  # M4
    import threading

    sid = stream(app, action="start", chunk_format="pcm", sample_rate_hz=16000, channels=1,
                 sample_format="s16le")["result"]["session_id"]
    part = base64.b64encode((np.ones(1600, dtype=np.int16) * 1000).tobytes()).decode()
    idx = []
    ts = [threading.Thread(target=lambda: idx.append(
        stream(app, action="push", session_id=sid, data=part)["result"]["chunk_index"])) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert sorted(idx) == list(range(8))
    s = app.streams._get(sid)
    assert s.buffer.stat().st_size == 8 * 1600 * 4 and s.chunks[-1][1] == 0.8
    stream(app, action="cancel", session_id=sid)
    with pytest.raises(AudioMCPError):  # a late push after close never writes into a deleted session
        app.streams.push(sid, part)


def test_push_waits_for_a_model_slot(app, speech_en):
    import threading

    x, sr = sf.read(speech_en, dtype="int16")
    sid = stream(app, action="start", chunk_format="pcm", sample_rate_hz=sr, channels=1,
                 sample_format="s16le")["result"]["session_id"]
    held = [app.slots.acquire(blocking=False) for _ in range(app.settings.max_concurrent_jobs)]
    assert all(held)  # every slot taken, e.g. by long async jobs
    done = threading.Event()
    t = threading.Thread(target=lambda: (stream(app, action="push", session_id=sid,
                                                data=base64.b64encode(x[:sr].tobytes()).decode()), done.set()))
    t.start()
    assert not done.wait(0.5)  # the push's model work queues behind the cap
    for _ in held:
        app.slots.release()
    assert done.wait(30)
    t.join()

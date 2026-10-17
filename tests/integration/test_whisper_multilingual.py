import subprocess

import pytest

from conftest import FIXTURES_DIR, fp
from audio_mcp.tools.handlers import transcribe_audio
from audio_mcp.tools.schemas import TranscribeRequest

pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("ignore::UserWarning")]


def test_en_es_code_switch(app, tmp_path):
    app.settings.backend_profile = "real"
    from audio_mcp.context import build_app_context

    real = build_app_context(app.settings)
    out = tmp_path / "en_es.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(FIXTURES_DIR / "speech_en.wav"), "-f", "lavfi", "-t", "1",
                    "-i", "anullsrc=r=16000:cl=mono", "-i", str(FIXTURES_DIR / "speech_es.ogg"),
                    "-filter_complex", "[0:a]aresample=16000,aformat=channel_layouts=mono[a];"
                    "[2:a]aresample=16000,aformat=channel_layouts=mono[c];[a][1:a][c]concat=n=3:v=0:a=1",
                    "-ar", "16000", "-ac", "1", str(out)], check=True)
    r = transcribe_audio(real, TranscribeRequest(source=fp(out), multilingual=True, translate_to_en=True))
    langs = [l["language"] for l in r["result"]["languages"]]
    assert langs[0] == "en" and "es" in langs
    es = [s for s in r["result"]["segments"] if s["language"] == "es"]
    assert es and all(s.get("translation") for s in es)

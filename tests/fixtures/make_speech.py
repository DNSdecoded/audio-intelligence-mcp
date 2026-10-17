"""Manual, Windows-only: regenerates speech_en / dialogue_2spk via SAPI (pyttsx3). Outputs are committed.

Run: uv run python tests/fixtures/make_speech.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pyttsx3
import soundfile as sf
import numpy as np

OUT = Path(__file__).parent / "audio"
EN = "We need to finish the report by Friday. The alarm test is scheduled for Monday morning."
TURNS = [
    ("David", "Did you send the budget numbers?"),
    ("Zira", "Yes, I sent them this morning."),
    ("David", "Great, then we can close the review."),
    ("Zira", "I will book the room for Thursday."),
]


def _say_once(voice, text, dst):
    engine = pyttsx3.init()
    vid = next(v.id for v in engine.getProperty("voices") if f"Microsoft {voice} Desktop" in v.name)
    engine.setProperty("voice", vid)
    engine.save_to_file(text, dst)
    engine.runAndWait()


def say(voice, text, dst: Path):
    # One process per utterance: a second runAndWait() in the same process hangs on SAPI.
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.wav"
        subprocess.run([sys.executable, __file__, "--say", voice, text, str(raw)], check=True, timeout=60)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst)], check=True)


def main():
    say("David", EN, OUT / "speech_en.wav")
    (OUT / "speech_en.txt").write_text(EN + "\n", encoding="utf-8")

    gap = np.zeros(int(0.4 * 16000), dtype=np.int16)
    parts, rttm, t = [], [], 0.0
    with tempfile.TemporaryDirectory() as td:
        for i, (spk, text) in enumerate(TURNS):
            f = Path(td) / f"t{i}.wav"
            say(spk, text, f)
            d, _ = sf.read(f, dtype="int16")
            rttm.append(f"SPEAKER dialogue_2spk 1 {t:.3f} {len(d) / 16000:.3f} <NA> <NA> {spk.lower()} <NA> <NA>")
            parts += [d, gap]
            t += len(d) / 16000 + 0.4
    sf.write(OUT / "dialogue_2spk.wav", np.concatenate(parts[:-1]), 16000, subtype="PCM_16")
    (OUT / "dialogue_2spk.rttm").write_text("\n".join(rttm) + "\n", encoding="utf-8")
    (OUT / "dialogue_2spk.txt").write_text("\n".join(f"{s.lower()}: {x}" for s, x in TURNS) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--say"]:
        _say_once(*sys.argv[2:5])
    else:
        main()

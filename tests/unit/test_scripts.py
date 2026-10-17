import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from conftest import FIXTURES_DIR

ROOT = Path(__file__).resolve().parents[2]


def env(tmp_path, *roots):
    e = {**os.environ, "AUDIO_MCP_DATA_DIR": str(tmp_path / "data"),
         "AUDIO_MCP_ALLOWED_ROOTS": json.dumps([str(r) for r in roots])}
    e.pop("AUDIO_MCP_CONFIG", None)
    return e


def run(args, e):
    out = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=e, cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_benchmark_runs(tmp_path):
    data = run(["scripts/benchmark.py", "--file", str(FIXTURES_DIR / "speech_en.wav"), "--tasks", "acoustic_features",
                "--repeat", "2", "--profile", "fake"], env(tmp_path, FIXTURES_DIR))
    names = [s["stage"] for s in data["stages"]]
    assert {"metadata", "features"} <= set(names) and "rtf" in data["stages"][0] and data["peak_rss_mb"] > 0
    assert data["repeats"] == 2 and data["duration_seconds"] > 4


def test_evaluate_wer_fake(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    shutil.copy(FIXTURES_DIR / "speech_en.wav", d)
    shutil.copy(FIXTURES_DIR / "speech_en.txt", d)
    rep = run(["scripts/evaluate.py", "wer", "--ref-dir", str(d), "--audio-dir", str(d), "--profile", "fake",
               "--out", str(tmp_path / "r.json")], env(tmp_path, d))
    assert rep["metric"] == "wer" and len(rep["per_file"]) == 1 and 0 <= rep["aggregate"]["wer"] <= 1
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8")) == rep


def test_evaluate_events_and_der_fake(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    shutil.copy(FIXTURES_DIR / "dialogue_2spk.wav", d)
    shutil.copy(FIXTURES_DIR / "dialogue_2spk.rttm", d)
    (d / "labels.csv").write_text("file,start,end,label\ndialogue_2spk.wav,0.0,2.0,Speech\n", encoding="utf-8")
    ev = run(["scripts/evaluate.py", "events", "--labels", str(d / "labels.csv"), "--audio-dir", str(d),
              "--profile", "fake"], env(tmp_path, d))
    assert {"precision", "recall", "f1"} <= ev["aggregate"].keys()
    der = run(["scripts/evaluate.py", "der", "--rttm-dir", str(d), "--audio-dir", str(d), "--profile", "fake"],
              env(tmp_path, d))
    assert 0 <= der["aggregate"]["der"]

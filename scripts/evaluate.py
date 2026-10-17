"""Accuracy evaluation against labelled data: WER, sound-event precision/recall, diarization error rate.

Usage:
  uv run python scripts/evaluate.py wer    --ref-dir D --audio-dir A        (pairs stem.txt with stem.<audio ext>)
  uv run python scripts/evaluate.py events --labels labels.csv --audio-dir A (CSV columns: file,start,end,label)
  uv run python scripts/evaluate.py der    --rttm-dir R --audio-dir A        (pairs stem.rttm with stem.<audio ext>)
Options: --profile real|fake (default real), --out report.json, --iou 0.5 (events), --collar 0.25 (der).
The audio directory must be inside allowed_roots (AUDIO_MCP_ALLOWED_ROOTS).
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from audio_mcp.config import load_settings
from audio_mcp.context import build_app_context
from audio_mcp.eval.metrics import der, event_prf, load_rttm, word_error_rate
from audio_mcp.library.index import SUPPORTED_EXTENSIONS
from audio_mcp.tools.handlers import analyze_audio, detect_audio_events, transcribe_audio
from audio_mcp.tools.schemas import AnalyzeRequest, DetectEventsRequest, TranscribeRequest


def _audio_for(stem: str, audio_dir: Path) -> Path | None:
    return next((p for p in sorted(audio_dir.glob(f"{stem}.*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS), None)


def _src(p: Path) -> dict:
    return {"type": "file_path", "path": str(p)}


def _check(resp: dict, p: Path) -> dict:
    if resp["status"] == "error":
        raise SystemExit(f"{p}: {resp['error']['code']}: {resp['error']['message']}")
    return resp


def eval_wer(app, a) -> dict:
    rows, edits, words = [], 0.0, 0
    for ref in sorted(Path(a.ref_dir).glob("*.txt")):
        audio = _audio_for(ref.stem, Path(a.audio_dir))
        if audio is None:
            continue
        text = ref.read_text(encoding="utf-8")
        hyp = _check(transcribe_audio(app, TranscribeRequest(source=_src(audio), async_mode="never")), audio)
        w = word_error_rate(text, hyp["result"]["text"])
        n = len(re.findall(r"\w+", text))
        rows.append({"file": audio.name, "wer": round(w, 4), "ref_words": n})
        edits, words = edits + w * n, words + n
    return {"metric": "wer", "per_file": rows,
            "aggregate": {"wer": round(edits / words, 4) if words else None, "files": len(rows)}}


def eval_events(app, a) -> dict:
    ref = defaultdict(list)
    with open(a.labels, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ref[r["file"]].append({"label": r["label"], "start": float(r["start"]), "end": float(r["end"])})
    rows, tot = [], {"tp": 0, "fp": 0, "fn": 0}
    for name, events in sorted(ref.items()):
        audio = Path(a.audio_dir) / name
        hyp = _check(detect_audio_events(app, DetectEventsRequest(source=_src(audio), async_mode="never")), audio)
        m = event_prf(events, hyp["result"]["events"], iou=a.iou)
        rows.append({"file": name, **m})
        for k in tot:
            tot[k] += m[k]
    p = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else 0.0
    r = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"metric": "events", "per_file": rows,
            "aggregate": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), **tot, "iou": a.iou}}


def eval_der(app, a) -> dict:
    rows, num, den = [], 0.0, 0.0
    for rttm in sorted(Path(a.rttm_dir).glob("*.rttm")):
        audio = _audio_for(rttm.stem, Path(a.audio_dir))
        if audio is None:
            continue
        resp = _check(analyze_audio(app, AnalyzeRequest(source=_src(audio), tasks=["speaker_diarization"],
                                                         async_mode="never")), audio)
        hyp = (resp["analysis"].get("speakers") or {}).get("turns", [])
        ref = load_rttm(rttm)
        m = der(ref, hyp, collar=a.collar)
        speech = sum(t["end"] - t["start"] for t in ref)
        rows.append({"file": audio.name, **{k: round(v, 4) for k, v in m.items()}})
        num, den = num + m["der"] * speech, den + speech
    return {"metric": "der", "per_file": rows,
            "aggregate": {"der": round(num / den, 4) if den else None, "files": len(rows), "collar": a.collar}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("wer", "events", "der"):
        p = sub.add_parser(name)
        p.add_argument("--audio-dir", required=True)
        p.add_argument("--profile", choices=["real", "fake"], default="real")
        p.add_argument("--out")
    sub.choices["wer"].add_argument("--ref-dir", required=True)
    sub.choices["events"].add_argument("--labels", required=True)
    sub.choices["events"].add_argument("--iou", type=float, default=0.5)
    sub.choices["der"].add_argument("--rttm-dir", required=True)
    sub.choices["der"].add_argument("--collar", type=float, default=0.25)
    a = ap.parse_args(argv)
    settings = load_settings()
    settings.backend_profile = a.profile
    app = build_app_context(settings)
    report = {"wer": eval_wer, "events": eval_events, "der": eval_der}[a.cmd](app, a)
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()

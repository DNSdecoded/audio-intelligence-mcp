"""Accuracy evaluation against labelled data: WER, sound-event precision/recall, diarization error rate.

Usage:
  uv run python scripts/evaluate.py wer    --ref-dir D --audio-dir A        (pairs stem.txt with stem.<audio ext>)
  uv run python scripts/evaluate.py events --labels labels.csv --audio-dir A (CSV columns: file,start,end,label)
  uv run python scripts/evaluate.py der    --rttm-dir R --audio-dir A        (pairs stem.rttm with stem.<audio ext>)
  uv run python scripts/evaluate.py quality --labels labels.csv --audio-dir A (CSV columns: file,issues; issues "A|B")
Options: --profile real|fake (default real), --out report.json, --iou 0.5 (events), --collar 0.25 (der).
The audio directory must be inside allowed_roots (AUDIO_MCP_ALLOWED_ROOTS).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from audio_mcp.config import load_settings
from audio_mcp.context import build_app_context
from audio_mcp.eval.metrics import _words, char_error_rate, der, event_prf, load_rttm, word_error_rate
from audio_mcp.library.index import SUPPORTED_EXTENSIONS
from audio_mcp.tools.handlers import analyze_audio, detect_audio_events, inspect_audio_quality, transcribe_audio
from audio_mcp.tools.schemas import AnalyzeRequest, DetectEventsRequest, InspectQualityRequest, TranscribeRequest


def _audio_for(stem: str, audio_dir: Path) -> Path | None:
    return next((p for p in sorted(audio_dir.glob(f"{stem}.*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS), None)


def _src(p: Path) -> dict:
    return {"type": "file_path", "path": str(p)}


def _check(resp: dict, p: Path) -> dict:
    if resp["status"] == "error":
        raise SystemExit(f"{p}: {resp['error']['code']}: {resp['error']['message']}")
    return resp


def eval_wer(app, a) -> dict:
    rate, unit = (char_error_rate, "cer") if a.unit == "char" else (word_error_rate, "wer")
    rows, edits, words, lang_ok = [], 0.0, 0, 0
    for ref in sorted(Path(a.ref_dir).glob("*.txt")):
        audio = _audio_for(ref.stem, Path(a.audio_dir))
        if audio is None:
            continue
        text = ref.read_text(encoding="utf-8")
        hyp = _check(transcribe_audio(app, TranscribeRequest(source=_src(audio), async_mode="never")), audio)
        w = rate(text, hyp["result"]["text"])
        n = len("".join(_words(text))) if unit == "cer" else len(_words(text))
        lang = hyp["result"].get("language")
        lang = lang.get("primary") if isinstance(lang, dict) else lang
        lang_ok += lang == a.language
        rows.append({"file": audio.name, unit: round(w, 4), "ref_units": n, "language": lang,
                     "hypothesis": hyp["result"]["text"]})
        edits, words = edits + w * n, words + n
    agg = {unit: round(edits / words, 4) if words else None, "files": len(rows)}
    if a.language:
        agg["language_accuracy"] = round(lang_ok / len(rows), 4) if rows else None
    return {"metric": unit, "per_file": rows, "aggregate": agg}


def eval_events(app, a) -> dict:
    ref = defaultdict(list)
    with open(a.labels, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ref[r["file"]].append({"label": r["label"], "start": float(r["start"]), "end": float(r["end"])})
    rows, tot, top1 = [], {"tp": 0, "fp": 0, "fn": 0}, 0
    for name, events in sorted(ref.items()):
        audio = Path(a.audio_dir) / name
        hyp = _check(detect_audio_events(app, DetectEventsRequest(source=_src(audio), async_mode="never")), audio)
        hyp_ev = hyp["result"]["events"]
        m = event_prf(events, hyp_ev, iou=a.iou)
        best = max(hyp_ev, key=lambda e: e.get("confidence", 0.0))["label"] if hyp_ev else None
        hit = best is not None and any(best.lower() in e["label"].lower().split("|") for e in events)
        top1 += hit
        rows.append({"file": name, **m, "top1": best, "top1_hit": hit})
        for k in tot:
            tot[k] += m[k]
    p = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else 0.0
    r = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"metric": "events", "per_file": rows,
            "aggregate": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), **tot, "iou": a.iou,
                          "top1_accuracy": round(top1 / len(rows), 4) if rows else None, "files": len(rows)}}


def eval_quality(app, a) -> dict:
    with open(a.labels, encoding="utf-8", newline="") as fh:
        ref = {r["file"]: {c for c in r["issues"].split("|") if c} for r in csv.DictReader(fh)}
    rows, per = [], defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for name, want in sorted(ref.items()):
        audio = Path(a.audio_dir) / name
        resp = _check(inspect_audio_quality(app, InspectQualityRequest(source=_src(audio), async_mode="never")), audio)
        got = {i["code"] for i in resp["result"]["issues"]} - {"MULTICHANNEL"}
        for c in want | got:
            per[c]["tp" if c in want and c in got else "fn" if c in want else "fp"] += 1
        rows.append({"file": name, "expected": sorted(want), "detected": sorted(got), "exact": want == got})
    codes = {c: {**v, "precision": round(v["tp"] / (v["tp"] + v["fp"]), 4) if v["tp"] + v["fp"] else None,
                 "recall": round(v["tp"] / (v["tp"] + v["fn"]), 4) if v["tp"] + v["fn"] else None}
             for c, v in sorted(per.items())}
    return {"metric": "quality", "per_file": rows, "per_issue": codes,
            "aggregate": {"exact_match": round(sum(r["exact"] for r in rows) / len(rows), 4) if rows else None,
                          "files": len(rows)}}


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
    for name in ("wer", "events", "der", "quality"):
        p = sub.add_parser(name)
        p.add_argument("--audio-dir", required=True)
        p.add_argument("--profile", choices=["real", "fake"], default="real")
        p.add_argument("--out")
    sub.choices["wer"].add_argument("--ref-dir", required=True)
    sub.choices["wer"].add_argument("--unit", choices=["word", "char"], default="word")
    sub.choices["wer"].add_argument("--language", help="expected language code; reports detection accuracy")
    sub.choices["events"].add_argument("--labels", required=True)
    sub.choices["events"].add_argument("--iou", type=float, default=0.5)
    sub.choices["der"].add_argument("--rttm-dir", required=True)
    sub.choices["der"].add_argument("--collar", type=float, default=0.25)
    sub.choices["quality"].add_argument("--labels", required=True)
    a = ap.parse_args(argv)
    out, sys.stdout = sys.stdout, sys.stderr  # model libraries print on load; keep stdout pure JSON
    settings = load_settings()
    settings.backend_profile = a.profile
    app = build_app_context(settings)
    report = {"wer": eval_wer, "events": eval_events, "der": eval_der, "quality": eval_quality}[a.cmd](app, a)
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    out.write(text + "\n")


if __name__ == "__main__":
    main()

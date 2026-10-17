"""Export an analysis payload to json, markdown, srt, vtt, csv or rttm files."""
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from audio_mcp.envelope import AudioMCPError, ErrorCode
from audio_mcp.export.redact import redact_analysis

EXPORT_FORMATS = ("json", "markdown", "srt", "vtt", "csv", "rttm")
_EXT = {"markdown": "md"}


def _clock(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _short(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _segments(a):
    return (a.get("transcription") or {}).get("segments", [])


def _srt(a):
    cues = []
    for i, s in enumerate(_segments(a), 1):
        spk = f"[{s['speaker_id']}] " if s.get("speaker_id") else ""
        cues.append(f"{i}\n{_clock(s['start'], ',')} --> {_clock(s['end'], ',')}\n{spk}{s['text'].strip()}\n")
    return "\n".join(cues)


def _vtt(a):
    cues = []
    for s in _segments(a):
        voice = f"<v {s['speaker_id']}>" if s.get("speaker_id") else ""
        cues.append(f"{_clock(s['start'], '.')} --> {_clock(s['end'], '.')}\n{voice}{s['text'].strip()}\n")
    return "WEBVTT\n\n" + "\n".join(cues)


def _csv(a):
    rows = [{"type": "speech", "id": s["id"], "start": s["start"], "end": s["end"],
             "speaker_id": s.get("speaker_id", ""), "label": "", "text": s["text"].strip(),
             "confidence": s.get("confidence", "")} for s in _segments(a)]
    rows += [{"type": "event", "id": e["id"], "start": e["start"], "end": e["end"], "speaker_id": "",
              "label": e["label"], "text": "", "confidence": e.get("confidence", "")}
             for e in a.get("sound_events", [])]
    buf = io.StringIO()
    w = csv.DictWriter(buf, ["type", "id", "start", "end", "speaker_id", "label", "text", "confidence"],
                       lineterminator="\n")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["start"], r["type"])))
    return buf.getvalue()


def _rttm(audio_id, a):
    turns = (a.get("speakers") or {}).get("turns", [])
    if not turns:
        raise AudioMCPError(ErrorCode.INVALID_INPUT, "no speaker turns; run with diarization")
    return "".join(f"SPEAKER {audio_id} 1 {t['start']:.3f} {t['end'] - t['start']:.3f} <NA> <NA> {t['speaker_id']} "
                   "<NA> <NA>\n" for t in turns)


def _markdown(audio_id, a):
    lines = [f"# Audio analysis: {audio_id}", ""]
    meta = a.get("audio_metadata") or {}
    if meta:
        lines += ["| field | value |", "|---|---|", *[f"| {k} | {v} |" for k, v in meta.items()], ""]
    summ = (a.get("summary") or {}).get("text")
    if summ:
        lines += ["## Summary", "", summ, ""]
    if _segments(a):
        lines += ["## Transcript", ""]
        for s in _segments(a):
            who = f" {s['speaker_id']}" if s.get("speaker_id") else ""
            lines.append(f"**[{_short(s['start'])}]{who}:** {s['text'].strip()}  ")
        lines.append("")
    if a.get("sound_events"):
        lines += ["## Sound events", "", "| start | end | label | confidence |", "|---|---|---|---|"]
        lines += [f"| {_short(e['start'])} | {_short(e['end'])} | {e['label']} | {e.get('confidence', '')} |"
                  for e in a["sound_events"]]
        lines.append("")
    issues = (a.get("audio_quality") or {}).get("issues", [])
    if issues:
        lines += ["## Quality issues", "", *[f"- **{i['code']}** ({i['severity']}): {i['message']}" for i in issues], ""]
    return "\n".join(lines)


def _out_path(out_dir: Path, audio_id: str, ext: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    p, n = out_dir / f"{audio_id}-{stamp}.{ext}", 1
    while p.exists():  # two exports in the same second
        p, n = out_dir / f"{audio_id}-{stamp}-{n}.{ext}", n + 1
    return p


def export_analysis(audio_id: str, analysis: dict, fmt: str, out_dir: Path, *, redact: bool,
                    extra_patterns=()) -> dict:
    if fmt not in EXPORT_FORMATS:
        raise AudioMCPError(ErrorCode.INVALID_INPUT, f"unsupported export format {fmt}")
    redactions = 0
    if redact:
        analysis, redactions = redact_analysis(analysis, extra_patterns)
    body = {"json": lambda: json.dumps(analysis, indent=2, ensure_ascii=False), "srt": lambda: _srt(analysis),
            "vtt": lambda: _vtt(analysis), "csv": lambda: _csv(analysis), "rttm": lambda: _rttm(audio_id, analysis),
            "markdown": lambda: _markdown(audio_id, analysis)}[fmt]()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _out_path(out_dir, audio_id, _EXT.get(fmt, fmt))
    data = body.encode("utf-8")
    path.write_bytes(data)
    return {"path": str(path), "format": fmt, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "redactions": redactions}

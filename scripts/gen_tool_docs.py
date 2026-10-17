"""Generate docs/tools.md (from the live server's tool list + docs/examples) and docs/configuration.md.

Usage:
  uv run python scripts/gen_tool_docs.py                   regenerate docs/tools.md and docs/configuration.md
  uv run python scripts/gen_tool_docs.py --check           exit 1 if either file is out of date
  uv run python scripts/gen_tool_docs.py --write-examples  re-run every tool (fake backends) into docs/examples/
"""
import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
DOCS, EXAMPLES = ROOT / "docs", ROOT / "docs" / "examples"
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
ORDER = ["analyze_audio", "transcribe_audio", "detect_audio_events", "extract_audio_features", "inspect_audio_quality",
         "get_audio_timeline", "ask_audio", "summarize_audio", "compare_audio", "search_audio_library",
         "analyze_audio_stream", "export_audio_analysis", "get_job"]


async def _with_server(samples: Path, data: Path, fn):
    env = {**os.environ, "AUDIO_MCP_BACKEND_PROFILE": "fake", "AUDIO_MCP_DATA_DIR": str(data),
           "AUDIO_MCP_ALLOWED_ROOTS": json.dumps([str(samples)])}
    env.pop("AUDIO_MCP_CONFIG", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "audio_mcp"], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        return await fn(s)


def _trim(x):
    """Keep examples readable: at most 3 list items and short strings."""
    if isinstance(x, dict):
        return {k: _trim(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_trim(v) for v in x[:3]]
    if isinstance(x, str) and len(x) > 200:
        return x[:200] + "..."
    return x


def _replace_paths(obj, mapping: dict[Path, str]):
    text = json.dumps(obj)
    for p, label in mapping.items():
        for form in {str(p), str(p).replace("\\", "/")}:
            text = text.replace(json.dumps(form)[1:-1], label)
    return json.loads(text)


async def _examples(s, samples: Path, outside: Path) -> dict:
    async def call(name, args):
        res = await s.call_tool(name, args)
        return json.loads(res.content[0].text)

    def src(name):
        return {"type": "file_path", "path": str(samples / name)}

    bad = {"type": "file_path", "path": str(outside)}
    ex = {}
    a = await call("analyze_audio", req := {"source": src("dialogue_2spk.wav"), "tasks": "auto", "goal": "meeting"})
    ex["analyze_audio"] = (req, a, {"source": bad})
    by_id = {"type": "audio_id", "audio_id": a["audio_id"]}
    plain = [("transcribe_audio", {"source": by_id, "diarize": True}),
             ("detect_audio_events", {"source": src("alarm.wav"), "categories": ["alarm"]}),
             ("extract_audio_features", {"source": src("speech_en.wav"), "feature_groups": ["level", "pitch"]}),
             ("inspect_audio_quality", {"source": src("sine.wav")}),
             ("get_audio_timeline", {"source": by_id}),
             ("ask_audio", {"source": by_id, "question": "When does someone mention the budget?"}),
             ("summarize_audio", {"source": by_id, "length": "short"}),
             ("compare_audio", {"sources": [src("sine.wav"), src("alarm.wav")]}),
             ("search_audio_library", {"action": "index", "paths": [str(samples)], "async_mode": "never"})]
    for name, req in plain:
        if name == "compare_audio":
            err = {"sources": [bad, bad]}
        elif name == "search_audio_library":
            err = {"action": "search"}
        else:
            err = {"source": bad, **({"question": "what happens?"} if name == "ask_audio" else {})}
        ex[name] = (req, await call(name, req), err)
    sid = (await call("analyze_audio_stream", {"action": "start", "chunk_format": "file"}))["result"]["session_id"]
    data = base64.b64encode((samples / "speech_en.wav").read_bytes()).decode()
    await call("analyze_audio_stream", {"action": "push", "session_id": sid, "data": data})
    req = {"action": "finish", "session_id": sid, "tasks": ["transcription"]}
    ex["analyze_audio_stream"] = (req, await call("analyze_audio_stream", req),
                                  {"action": "status", "session_id": "str_000000000000"})
    req = {"audio_id": a["audio_id"], "format": "srt"}
    ex["export_audio_analysis"] = (req, await call("export_audio_analysis", req),
                                   {"audio_id": "aud_0000000000000000", "format": "json"})
    job = await call("transcribe_audio", {"source": src("speech_en.wav"), "async_mode": "always"})
    await asyncio.sleep(2)
    req = {"job_id": job["job_id"]}
    ex["get_job"] = (req, await call("get_job", req), {"job_id": "job_000000000000"})
    out = {}
    for name, (req, ok, err_req) in ex.items():
        out[name] = {"request": req, "success": ok, "error_request": err_req, "error": await call(name, err_req)}
    return out


def write_examples():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        samples, data = tmp / "samples", tmp / "data"
        samples.mkdir()
        sys.path.insert(0, str(ROOT / "tests"))
        from fixtures.synth import make_synthetic

        synth = make_synthetic(tmp / "synth")
        for p in (FIXTURES / "speech_en.wav", FIXTURES / "dialogue_2spk.wav", synth["alarm"], synth["sine"]):
            shutil.copy(p, samples / p.name)
        outside = tmp / "outside.wav"
        shutil.copy(synth["sine"], outside)
        ex = asyncio.run(_with_server(samples, data, lambda s: _examples(s, samples, outside)))
        EXAMPLES.mkdir(parents=True, exist_ok=True)
        mapping = {outside: "./outside_allowed_roots/x.wav", samples: "./samples", data: "<data_dir>"}
        for name, e in ex.items():
            e = _replace_paths(_trim(e), mapping)
            (EXAMPLES / f"{name}.json").write_text(json.dumps(e, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8", newline="\n")


def _tools_md(tools) -> str:
    by = {t.name: t for t in tools}
    lines = ["# Tool reference", "", "Generated by `scripts/gen_tool_docs.py` from the running server; do not edit.",
             "Every tool returns the common envelope described in the README (`status`, `audio_id`, `result` or",
             "`analysis`, `warnings`, `errors`, `processing`). Examples come from the `fake` backend profile and are",
             "trimmed to 3 list items.", ""]
    lines += [f"- [`{n}`](#{n})" for n in ORDER if n in by] + [""]
    for n in ORDER:
        if n not in by:
            continue
        t = by[n]
        lines += [f"## {n}", "", t.description or "", "", "### Input schema", "", "```json",
                  json.dumps(t.inputSchema, indent=2), "```", ""]
        f = EXAMPLES / f"{n}.json"
        if f.exists():
            e = json.loads(f.read_text(encoding="utf-8"))
            for title, key in (("Example request", "request"), ("Example response", "success"),
                               ("Example error request", "error_request"), ("Example error response", "error")):
                lines += [f"### {title}", "", "```json", json.dumps(e[key], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def _config_md() -> str:
    from audio_mcp.config import Settings

    rows = {}
    for line in (ROOT / "config.example.toml").read_text(encoding="utf-8").splitlines():
        m = re.match(r"#?\s*([a-z_]+)\s*=\s*(.*?)\s+#\s*(.*)$", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3))
    lines = ["# Configuration", "", "Generated by `scripts/gen_tool_docs.py` from `config.example.toml`; do not edit.",
             "", "Settings load in this order (later wins): defaults, the TOML file named by `AUDIO_MCP_CONFIG`, then",
             "environment variables `AUDIO_MCP_<NAME>` (list values as JSON, e.g. "
             "`AUDIO_MCP_ALLOWED_ROOTS='[\"/data\"]'`).",
             "", "| Setting | Env var | Type | Example / default | Meaning |", "|---|---|---|---|---|"]
    for name, field in Settings.model_fields.items():
        example, meaning = rows.get(name, ("", ""))
        a = field.annotation
        typ = (a.__name__ if isinstance(a, type) else str(a)).replace("typing.", "").replace("pathlib.", "")
        typ = typ.replace("|", "\\|")
        lines.append(f"| `{name}` | `AUDIO_MCP_{name.upper()}` | `{typ}` | `{example}` | {meaning} |")
    return "\n".join(lines) + "\n"


def render() -> dict[Path, str]:
    with tempfile.TemporaryDirectory() as t:
        tools = asyncio.run(_with_server(Path(t), Path(t) / "data", lambda s: s.list_tools())).tools
    return {DOCS / "tools.md": _tools_md(tools) + "\n", DOCS / "configuration.md": _config_md()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-examples", action="store_true")
    a = ap.parse_args()
    if a.write_examples:
        write_examples()
    stale = []
    for path, text in render().items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            stale.append(path.name)
            if not a.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="\n")
    if a.check and stale:
        print(f"out of date: {', '.join(stale)} (run scripts/gen_tool_docs.py)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

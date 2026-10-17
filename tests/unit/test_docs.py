import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ["analyze_audio", "transcribe_audio", "detect_audio_events", "extract_audio_features", "inspect_audio_quality",
         "get_audio_timeline", "ask_audio", "summarize_audio", "compare_audio", "search_audio_library",
         "analyze_audio_stream", "export_audio_analysis", "get_job"]


def test_generated_docs_in_sync():
    r = subprocess.run([sys.executable, "scripts/gen_tool_docs.py", "--check"], cwd=ROOT, capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]


def test_capabilities_sections():
    caps = (ROOT / "docs" / "capabilities.md").read_text(encoding="utf-8")
    assert all(h in caps for h in ["## Implemented", "## Partial", "## Future"])


def test_every_tool_documented_with_examples():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tools_md = (ROOT / "docs" / "tools.md").read_text(encoding="utf-8")
    for t in TOOLS:
        assert f"`{t}`" in readme and f"## {t}" in tools_md, t
        assert (ROOT / "docs" / "examples" / f"{t}.json").exists(), t


def test_readme_links_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\(((?:docs|tests)/[^)#]+)\)", readme):
        assert (ROOT / link).exists(), link

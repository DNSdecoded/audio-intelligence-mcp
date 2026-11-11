"""The routing skill must not drift from the tool schemas (same idea as docs/tools.md being generated)."""
import json
import re
from pathlib import Path
from typing import get_args

import pytest

from audio_mcp import server
from audio_mcp.tools import schemas as S

SKILL = Path(__file__).parents[2] / "skills" / "audio-intelligence" / "SKILL.md"
MODELS = {
    "analyze_audio": S.AnalyzeRequest, "transcribe_audio": S.TranscribeRequest,
    "detect_audio_events": S.DetectEventsRequest, "extract_audio_features": S.FeaturesRequest,
    "inspect_audio_quality": S.InspectQualityRequest, "get_audio_timeline": S.TimelineRequest,
    "ask_audio": S.AskRequest, "summarize_audio": S.SummarizeRequest, "compare_audio": S.CompareRequest,
    "search_audio_library": S.SearchLibraryRequest, "get_job": S.GetJobRequest,
    "analyze_audio_stream": S.StreamRequest, "export_audio_analysis": S.ExportRequest,
}
text = SKILL.read_text(encoding="utf-8")


def test_every_tool_is_covered():
    import anyio

    registered = {t.name for t in anyio.run(server.mcp.list_tools)}
    assert registered == set(MODELS)  # a new tool must get a schema mapping here and a row in the skill
    assert all(f"`{name}`" in text for name in registered)


def test_every_category_listed():
    assert all(f"`{c}`" in text for c in get_args(S.CategoryName))
    assert all(f"`{g}`" in text for g in get_args(S.FeatureGroup))


CALLS = re.findall(r"^(\w+) (\{.*\})$", text, re.M)


def test_has_worked_sequences():
    assert {n for n, _ in CALLS} >= {"analyze_audio", "ask_audio", "export_audio_analysis", "compare_audio",
                                     "detect_audio_events", "search_audio_library", "get_job"}


@pytest.mark.parametrize("name,args", CALLS)
def test_example_call_validates(name, args):
    MODELS[name](**json.loads(args))


def test_frontmatter_triggers():
    head = text.split("---")[1]
    assert re.search(r"^name: audio-intelligence$", head, re.M)
    desc = re.search(r"^description: (.+)$", head, re.M).group(1)
    assert ".wav" in desc and "subtitles" in desc and "Not for generating" in desc and "audio-intel " not in desc


def test_plugin_manifest_matches_skill_and_package():
    root = SKILL.parents[2]
    plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((root / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert plugin["name"] == SKILL.parent.name == market["plugins"][0]["name"]
    assert "version" not in market["plugins"][0]  # the manifest's version wins; keep one source
    cmd = plugin["mcpServers"]["audio-intelligence"]
    assert cmd["command"] == "uvx" and "audio-mcp" in cmd["args"]
    import tomllib
    proj = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(a.startswith(proj["name"]) for a in cmd["args"]) and "audio-mcp" in proj["scripts"]

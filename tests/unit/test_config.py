import os

import pytest

from audio_mcp.config import load_settings


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # keep dir creation and env out of the real user profile
    for k in [k for k in os.environ if k.startswith("AUDIO_MCP_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("AUDIO_MCP_DATA_DIR", str(tmp_path / "data"))


def test_defaults(monkeypatch):
    monkeypatch.delenv("AUDIO_MCP_CONFIG", raising=False)
    s = load_settings()
    assert s.allow_url_fetch is False and s.retention == "ttl" and s.retention_ttl_hours == 24
    assert s.llm_provider == "none" and s.backend_profile == "real"


def test_toml_then_env_precedence(tmp_path, monkeypatch):
    p = tmp_path / "c.toml"
    p.write_text('max_upload_mb = 7\nasr_model = "base"\n')
    monkeypatch.setenv("AUDIO_MCP_ASR_MODEL", "tiny")
    s = load_settings(p)
    assert s.max_upload_mb == 7 and s.asr_model == "tiny"


def test_llm_requires_model(monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError):
        load_settings()


def test_ollama_default_base_url(monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AUDIO_MCP_LLM_MODEL", "llama3")
    assert load_settings().llm_base_url == "http://localhost:11434/v1"


def test_resolved_dirs_created_and_list_env_json(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_ALLOWED_ROOTS", f'["{tmp_path.as_posix()}"]')
    s = load_settings()
    assert s.allowed_roots == [tmp_path]
    assert (tmp_path / "data" / "models").is_dir()


def test_missing_config_file_raises(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.toml")
    monkeypatch.setenv("AUDIO_MCP_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(FileNotFoundError):
        load_settings()

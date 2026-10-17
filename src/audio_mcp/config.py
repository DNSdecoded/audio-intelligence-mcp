import os
from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_MCP_", extra="ignore")

    allowed_roots: list[Path] = [Path.home()]
    data_dir: Path = platformdirs.user_data_path("audio-mcp")
    temp_dir: Path | None = None
    export_dir: Path | None = None
    models_dir: Path | None = None

    max_file_mb: int = 500
    max_upload_mb: int = 50
    max_duration_seconds: float = 14400

    allow_url_fetch: bool = False
    url_timeout_seconds: float = 30

    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None

    backend_profile: Literal["real", "fake"] = "real"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    asr_model: str = "small"
    asr_compute_type: str = "int8"

    diarization_backend: Literal["ecapa", "pyannote"] = "ecapa"
    diarization_threshold: float = 0.55
    max_speakers: int = 8

    event_threshold: float = 0.3
    low_confidence_threshold: float = 0.5
    speech_ratio_threshold: float = 0.2

    auto_preprocess: bool = False

    async_threshold_seconds: float = 600
    max_concurrent_jobs: int = 1
    stage_timeout_seconds: float = 1800

    retention: Literal["none", "ttl", "keep"] = "ttl"
    retention_ttl_hours: float = 24
    library_retention: Literal["keep", "ttl"] = "keep"

    log_level: str = "INFO"
    log_sensitive: bool = False
    log_file: Path | None = None

    llm_provider: Literal["none", "anthropic", "openai_compatible", "ollama"] = "none"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key_env: str = "AUDIO_MCP_LLM_API_KEY"
    llm_timeout_seconds: float = 60

    redact_outputs: bool = False
    redact_extra_patterns: list[str] = []

    @model_validator(mode="after")
    def _check_llm(self) -> "Settings":
        if self.llm_provider != "none" and not self.llm_model:
            raise ValueError("llm_model is required when llm_provider is not 'none'")
        if self.llm_provider == "ollama" and not self.llm_base_url:
            self.llm_base_url = "http://localhost:11434/v1"
        return self

    @property
    def resolved_temp_dir(self) -> Path:
        return self.temp_dir or self.data_dir / "tmp"

    @property
    def resolved_export_dir(self) -> Path:
        return self.export_dir or self.data_dir / "exports"

    @property
    def resolved_models_dir(self) -> Path:
        return self.models_dir or self.data_dir / "models"


def load_settings(config_path: Path | None = None) -> Settings:
    """Precedence: env > TOML (arg, else AUDIO_MCP_CONFIG) > defaults."""
    path = config_path or (Path(p) if (p := os.environ.get("AUDIO_MCP_CONFIG")) else None)

    if path and not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    class _Settings(Settings):
        @classmethod
        def settings_customise_sources(
            cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
            if path:
                sources.append(TomlConfigSettingsSource(settings_cls, toml_file=path))
            return tuple(sources)

    s = _Settings()
    for d in (s.resolved_temp_dir, s.resolved_export_dir, s.resolved_models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s

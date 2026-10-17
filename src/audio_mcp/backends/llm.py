"""Optional text-only LLM backends. Audio never leaves the machine; only evidence text is sent."""
import ipaddress
import os
import socket
import threading
from urllib.parse import urlparse

from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo

_ANTHROPIC_URL = "https://api.anthropic.com"
# models that accept server-side refusal fallbacks ("default" routing by refusal category)
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_MAX_TOKENS = 4096


def is_loopback(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        addrs = {i[4][0].split("%")[0] for i in socket.getaddrinfo(host, None)}
        return bool(addrs) and all(ipaddress.ip_address(a).is_loopback for a in addrs)
    except (OSError, ValueError):
        return False


class NoLLM:
    info = ModelInfo(stage="llm", name="none", version="0")
    is_remote = False

    def available(self):
        return False, "llm_provider is none"

    def load(self):
        raise AudioMCPError(ErrorCode.LLM_DISABLED, "llm_provider is none")

    def complete(self, system, prompt):
        raise AudioMCPError(ErrorCode.LLM_DISABLED, "llm_provider is none")


class AnthropicLLM:
    """Claude via the official anthropic SDK (optional `llm` extra)."""

    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings.llm_base_url or _ANTHROPIC_URL
        self.is_remote = not is_loopback(self.base_url)
        self.info = ModelInfo(stage="llm", name="anthropic", version=settings.llm_model, local=not self.is_remote)
        self._client = None
        self._lock = threading.Lock()

    def available(self):
        from audio_mcp.backends.base import _importable

        if not _importable("anthropic"):
            return False, "anthropic not installed (uv sync --extra llm)"
        return True, None

    def load(self):
        with self._lock:
            if self._client is None:
                import anthropic

                key = os.environ.get(self.settings.llm_api_key_env) or None  # None: SDK resolves its own credentials
                self._client = anthropic.Anthropic(api_key=key, base_url=self.settings.llm_base_url,
                                                   timeout=self.settings.llm_timeout_seconds)

    def complete(self, system: str, prompt: str) -> str:
        import anthropic

        self.load()
        kw = dict(model=self.settings.llm_model, max_tokens=_MAX_TOKENS, system=system,
                  messages=[{"role": "user", "content": prompt}])
        try:
            if self.settings.llm_model in _FALLBACK_MODELS:
                resp = self._client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                         fallbacks="default", **kw)
            else:
                resp = self._client.messages.create(**kw)
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            raise AudioMCPError(ErrorCode.STAGE_FAILED, f"LLM request failed: {type(e).__name__}", retryable=True) from e
        except anthropic.APIStatusError as e:
            raise AudioMCPError(ErrorCode.STAGE_FAILED, f"LLM request failed: HTTP {e.status_code}",
                                retryable=e.status_code >= 500) from e
        if resp.stop_reason == "refusal":
            raise AudioMCPError(ErrorCode.STAGE_FAILED, "LLM declined the request")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatibleLLM:
    """Any /chat/completions endpoint (Ollama, vLLM, LM Studio, ...) via httpx."""

    def __init__(self, settings):
        self.settings = settings
        self.base_url = (settings.llm_base_url or "").rstrip("/")
        self.is_remote = not is_loopback(self.base_url)
        self.info = ModelInfo(stage="llm", name=settings.llm_provider, version=settings.llm_model,
                              local=not self.is_remote)

    def available(self):
        return (True, None) if self.base_url else (False, "llm_base_url is not set")

    def load(self):
        pass

    def complete(self, system: str, prompt: str) -> str:
        import httpx

        headers = {}
        key = os.environ.get(self.settings.llm_api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {"model": self.settings.llm_model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                           timeout=self.settings.llm_timeout_seconds)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except httpx.HTTPStatusError as e:
            raise AudioMCPError(ErrorCode.STAGE_FAILED, f"LLM request failed: HTTP {e.response.status_code}",
                                retryable=e.response.status_code >= 500) from e
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise AudioMCPError(ErrorCode.STAGE_FAILED, f"LLM request failed: {type(e).__name__}",
                                retryable=True) from e


def build_llm(settings):
    if settings.llm_provider == "none":
        return NoLLM()
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    return OpenAICompatibleLLM(settings)

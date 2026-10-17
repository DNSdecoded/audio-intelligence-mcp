import base64
import binascii
import ipaddress
import os
import re
import socket
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from audio_mcp.config import Settings
from audio_mcp.envelope import AudioMCPError, ErrorCode


class FilePathSource(BaseModel):
    type: Literal["file_path"]
    path: str


class Base64Source(BaseModel):
    type: Literal["base64"]
    data: str
    filename_hint: str | None = None


class UrlSource(BaseModel):
    type: Literal["url"]
    url: str


class AudioIdSource(BaseModel):
    type: Literal["audio_id"]
    audio_id: str = Field(pattern=r"^aud_[0-9a-f]{16}$")


AudioSource = Annotated[
    FilePathSource | Base64Source | UrlSource | AudioIdSource, Field(discriminator="type")
]


@dataclass
class MaterializedSource:
    path: Path
    original_name: str
    is_temp: bool


_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}")


def _norm(p: Path) -> Path:
    if sys.platform == "win32":
        return Path(os.path.normcase(str(p)))
    if sys.platform == "darwin":
        return Path(str(p).casefold())
    return p


def check_path_allowed(path: str | Path, roots: list[Path]) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise AudioMCPError(ErrorCode.FILE_NOT_FOUND, "file not found") from None
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioMCPError(ErrorCode.INVALID_INPUT, f"invalid path ({type(exc).__name__})") from None
    norm = _norm(resolved)
    # is_relative_to is component-wise, so /a/bc is not inside /a/b.
    if not any(norm.is_relative_to(_norm(Path(r).resolve())) for r in roots):
        raise AudioMCPError(ErrorCode.PATH_NOT_ALLOWED, "path is outside the allowed roots")
    if resolved.is_dir():
        raise AudioMCPError(ErrorCode.INVALID_INPUT, "path is a directory, not a file")
    return resolved


def _check_size(size: int, limit_mb: float, what: str = "file") -> None:
    if size > limit_mb * 2**20:
        raise AudioMCPError(ErrorCode.FILE_TOO_LARGE, f"{what} exceeds {limit_mb} MB limit")


def _from_base64(src: Base64Source, settings: Settings) -> MaterializedSource:
    if len(src.data) * 3 / 4 > settings.max_upload_mb * 2**20:
        raise AudioMCPError(
            ErrorCode.FILE_TOO_LARGE, f"upload exceeds {settings.max_upload_mb} MB limit"
        )
    try:
        raw = base64.b64decode(src.data, validate=True)
    except (binascii.Error, ValueError):
        raise AudioMCPError(ErrorCode.INVALID_INPUT, "data is not valid base64") from None
    hint = src.filename_hint or ""
    suffix = Path(hint.replace("\\", "/")).suffix
    ext = suffix if _EXT_RE.fullmatch(suffix) else ".bin"
    out = settings.resolved_temp_dir / f"{uuid.uuid4().hex}{ext}"
    out.write_bytes(raw)
    return MaterializedSource(out, Path(hint.replace("\\", "/")).name or out.name, True)


def _blocked(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return (
        a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
        or a.is_multicast or a.is_unspecified
    )


def _from_url(src: UrlSource, settings: Settings) -> MaterializedSource:
    if not settings.allow_url_fetch:
        raise AudioMCPError(ErrorCode.URL_FETCH_DISABLED, "URL fetching is disabled")
    parsed = urlparse(src.url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise AudioMCPError(ErrorCode.INVALID_INPUT, "only http(s) URLs are supported")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except (socket.gaierror, UnicodeError, ValueError):
        raise AudioMCPError(ErrorCode.URL_FETCH_FAILED, "could not resolve host", retryable=True) from None
    if any(_blocked(i[4][0]) for i in infos):
        raise AudioMCPError(ErrorCode.URL_FETCH_FAILED, "blocked address")
    # shortcut: httpx re-resolves the host (DNS-rebinding window); pin the vetted IP if this threat matters.
    ext = Path(parsed.path).suffix
    ext = ext if _EXT_RE.fullmatch(ext) else ".bin"
    out = settings.resolved_temp_dir / f"{uuid.uuid4().hex}{ext}"
    try:
        with httpx.stream(
            "GET", src.url, follow_redirects=False, timeout=settings.url_timeout_seconds
        ) as r:
            if r.status_code != 200:
                raise AudioMCPError(
                    ErrorCode.URL_FETCH_FAILED, f"unexpected HTTP status {r.status_code}", retryable=True
                )
            total = 0
            with out.open("wb") as fh:
                for chunk in r.iter_bytes():
                    total += len(chunk)
                    _check_size(total, settings.max_file_mb)
                    fh.write(chunk)
    except AudioMCPError:
        out.unlink(missing_ok=True)
        raise
    except httpx.HTTPError as exc:
        out.unlink(missing_ok=True)
        raise AudioMCPError(
            ErrorCode.URL_FETCH_FAILED, f"fetch failed ({type(exc).__name__})", retryable=True
        ) from None
    return MaterializedSource(out, Path(parsed.path).name or out.name, True)


def materialize_source(
    src: FilePathSource | Base64Source | UrlSource, settings: Settings
) -> MaterializedSource:
    if isinstance(src, FilePathSource):
        p = check_path_allowed(src.path, settings.allowed_roots)
        _check_size(p.stat().st_size, settings.max_file_mb)
        return MaterializedSource(p, p.name, False)
    if isinstance(src, Base64Source):
        return _from_base64(src, settings)
    return _from_url(src, settings)

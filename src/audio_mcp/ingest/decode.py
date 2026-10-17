"""Probe and decode audio via ffprobe/ffmpeg (arg lists only, never a shell)."""
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from audio_mcp.config import Settings
from audio_mcp.envelope import AudioMCPError, ErrorCode

_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_ffmpeg(settings: Settings) -> tuple[Path, Path]:
    ff = settings.ffmpeg_path or shutil.which("ffmpeg")
    fp = settings.ffprobe_path or shutil.which("ffprobe")
    if not ff or not fp:
        raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, "ffmpeg/ffprobe not found; see README install steps")
    return Path(ff), Path(fp)


def _run(args: list, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([str(a) for a in args], capture_output=True, timeout=timeout, creationflags=_FLAGS)
    except subprocess.TimeoutExpired as exc:
        raise AudioMCPError(ErrorCode.STAGE_FAILED, f"{Path(str(args[0])).name} timed out", retryable=True) from exc


@dataclass
class ProbeInfo:
    format_name: str
    codec: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    bit_depth: int | None
    size_bytes: int

    def to_dict(self) -> dict:
        return {
            "format": self.format_name,
            "codec": self.codec,
            "duration_seconds": self.duration_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "size_bytes": self.size_bytes,
        }


def probe(path: Path, settings: Settings) -> ProbeInfo:
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        raise AudioMCPError(ErrorCode.EMPTY_AUDIO, "file is empty")
    _, ffprobe = find_ffmpeg(settings)
    r = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], 30)
    if r.returncode != 0:
        raise AudioMCPError(ErrorCode.CORRUPT_AUDIO, "ffprobe could not read the file")
    try:
        data = json.loads(r.stdout)
    except ValueError as exc:
        raise AudioMCPError(ErrorCode.CORRUPT_AUDIO, "ffprobe returned unreadable output") from exc
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioMCPError(ErrorCode.UNSUPPORTED_AUDIO_FORMAT, "no audio stream found")
    s, fmt = streams[0], data.get("format", {})
    try:
        duration = float(s.get("duration") or fmt.get("duration") or 0)
    except ValueError:
        duration = 0.0
    if duration <= 0:
        raise AudioMCPError(ErrorCode.EMPTY_AUDIO, "audio has no duration")
    if duration > settings.max_duration_seconds:
        raise AudioMCPError(
            ErrorCode.DURATION_TOO_LONG,
            f"audio is {duration:.1f}s; limit is {settings.max_duration_seconds:.1f}s",
        )
    bits = int(s.get("bits_per_raw_sample") or 0) or int(s.get("bits_per_sample") or 0) or None
    return ProbeInfo(
        format_name=fmt.get("format_name", ""),
        codec=s.get("codec_name", ""),
        duration_seconds=duration,
        sample_rate_hz=int(s.get("sample_rate") or 0),
        channels=int(s.get("channels") or 0),
        bit_depth=bits,
        size_bytes=size,
    )


@dataclass
class DecodedAudio:
    native: np.ndarray  # float32 (channels, n)
    native_sr: int
    mono16k: np.ndarray  # float32 (n16,)

    @property
    def duration(self) -> float:
        return self.native.shape[1] / self.native_sr


def decode(path: Path, info: ProbeInfo, settings: Settings) -> DecodedAudio:
    ffmpeg, _ = find_ffmpeg(settings)
    timeout = max(60.0, 2 * info.duration_seconds)

    def run(*opts) -> np.ndarray:
        r = _run([ffmpeg, "-v", "error", "-xerror", "-i", path, *opts, "-f", "f32le", "-acodec", "pcm_f32le", "-"], timeout)
        if r.returncode != 0:
            raise AudioMCPError(ErrorCode.CORRUPT_AUDIO, "ffmpeg could not decode the file")
        n = len(r.stdout) // 4
        if n == 0:
            raise AudioMCPError(ErrorCode.EMPTY_AUDIO, "decoded audio is empty")
        return np.frombuffer(r.stdout[: n * 4], dtype="<f4").astype(np.float32)

    ch = max(info.channels, 1)
    flat = run("-ac", str(ch), "-ar", str(info.sample_rate_hz))
    native = flat[: len(flat) // ch * ch].reshape(-1, ch).T.copy()
    mono = run("-ac", "1", "-ar", "16000")
    return DecodedAudio(native=native, native_sr=info.sample_rate_hz, mono16k=mono)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = np.gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=-1).astype(np.float32)

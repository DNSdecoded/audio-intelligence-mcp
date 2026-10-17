"""Retention: TTL sweeps of stored audio, cached results, stream sessions, jobs and (optionally) the library."""
import os
import shutil
import time
from pathlib import Path

from audio_mcp.envelope import AudioMCPError


_STALE_SECONDS = 24 * 3600


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would send CTRL_C_EVENT on Windows
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists, owned by someone else
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove(p: Path) -> int:
    if p.is_dir():
        n = sum(1 for f in p.rglob("*") if f.is_file())
        shutil.rmtree(p, ignore_errors=True)
        return n
    p.unlink(missing_ok=True)
    return 1


def cleanup_temp(settings, *, own: bool = False) -> int:
    """own=True (shutdown): remove this instance's temp dir only. Otherwise (startup): remove temp dirs of
    instances that are gone (owner pid dead, or untouched for 24 h) and loose legacy files; never a live
    instance's files."""
    if own:
        d = settings.resolved_temp_dir / settings._instance_tag
        return _remove(d) if d.exists() else 0
    t = settings.resolved_temp_dir
    if not t.exists():
        return 0
    n, now = 0, time.time()
    for p in t.iterdir():
        if p.name == settings._instance_tag:
            continue
        parts = p.name.split("_")
        if p.is_dir() and len(parts) == 3 and parts[0] == "proc" and parts[1].isdigit():
            try:
                stale = not _pid_alive(int(parts[1])) or now - p.stat().st_mtime > _STALE_SECONDS
            except OSError:
                continue
            if not stale:
                continue
        n += _remove(p)
    return n


def sweep(app, *, now: float | None = None, final: bool = False) -> dict:
    s = app.settings
    now = time.time() if now is None else now
    cutoff = now - s.retention_ttl_hours * 3600
    out = {"audio_deleted": 0, "results_deleted": 0, "sessions_expired": 0, "jobs_pruned": 0,
           "library_rows_deleted": 0}
    if s.retention == "ttl" or (s.retention == "none" and final):
        ids = app.store.list_ids()
        if s.retention == "none":  # final sweep: only audio this instance ingested; others may share data_dir
            ids = [a for a in ids if a in app.store.created_ids]
        for aid in ids:
            try:
                expired = s.retention == "none" or app.store.get(aid, touch=False).last_used < cutoff
            except AudioMCPError:  # removed concurrently by another instance
                continue
            if expired:
                app.store.delete(aid)
                app.cache.clear(aid)
                out["audio_deleted"] += 1
                out["results_deleted"] += 1
        results = app.cache.root
        if s.retention == "ttl" and results.exists():  # orphaned result dirs (audio already gone)
            for d in results.iterdir():
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    out["results_deleted"] += 1
    if app.streams is not None:
        out["sessions_expired"] = app.streams.expire()
    if app.jobs is not None:
        out["jobs_pruned"] = app.jobs.prune(cutoff)
    if s.library_retention == "ttl" and app.library is not None:
        out["library_rows_deleted"] = app.library.remove_older_than(cutoff)
    return out

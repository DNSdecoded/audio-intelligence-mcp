"""Retention: TTL sweeps of stored audio, cached results, stream sessions, jobs and (optionally) the library."""
import shutil
import time


def cleanup_temp(settings) -> int:
    n = 0
    t = settings.resolved_temp_dir
    if not t.exists():
        return 0
    for p in t.iterdir():
        if p.is_dir():
            n += sum(1 for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
            n += 1
    return n


def sweep(app, *, now: float | None = None, final: bool = False) -> dict:
    s = app.settings
    now = time.time() if now is None else now
    cutoff = now - s.retention_ttl_hours * 3600
    out = {"audio_deleted": 0, "results_deleted": 0, "sessions_expired": 0, "jobs_pruned": 0,
           "library_rows_deleted": 0}
    if s.retention == "ttl" or (s.retention == "none" and final):
        for aid in app.store.list_ids():
            if s.retention == "none" or app.store.get(aid, touch=False).last_used < cutoff:
                app.store.delete(aid)
                app.cache.clear(aid)
                out["audio_deleted"] += 1
                out["results_deleted"] += 1
        results = app.cache.root
        if results.exists():  # orphaned result dirs (audio already gone)
            for d in results.iterdir():
                if d.is_dir() and (s.retention == "none" or d.stat().st_mtime < cutoff):
                    shutil.rmtree(d, ignore_errors=True)
                    out["results_deleted"] += 1
    if app.streams is not None:
        out["sessions_expired"] = app.streams.expire()
    if app.jobs is not None:
        out["jobs_pruned"] = app.jobs.prune(cutoff)
    if s.library_retention == "ttl" and app.library is not None:
        out["library_rows_deleted"] = app.library.remove_older_than(cutoff)
    return out

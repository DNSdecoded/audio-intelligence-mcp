"""Per-stage latency, real-time factor and peak memory for one audio file.

Usage: uv run python scripts/benchmark.py --file F [--tasks acoustic_features transcription ...] [--repeat 3]
                                          [--profile real|fake]
Settings come from AUDIO_MCP_* env vars / AUDIO_MCP_CONFIG; the file's folder must be in allowed_roots.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import psutil

from audio_mcp.config import load_settings
from audio_mcp.context import build_app_context
from audio_mcp.ingest.sources import FilePathSource
from audio_mcp.pipeline.planner import TASK_TO_STAGES, topo_order
from audio_mcp.pipeline.runner import run_analysis


class PeakRSS:
    def __init__(self, interval=0.05):
        self.proc, self.interval, self.peak = psutil.Process(os.getpid()), interval, 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", required=True)
    ap.add_argument("--tasks", nargs="+", default=["acoustic_features"], choices=sorted(TASK_TO_STAGES))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--profile", choices=["real", "fake"], default="real")
    a = ap.parse_args(argv)
    out, sys.stdout = sys.stdout, sys.stderr  # model libraries print on load; keep stdout pure JSON
    settings = load_settings()
    settings.backend_profile = a.profile
    app = build_app_context(settings)
    src = FilePathSource(type="file_path", path=a.file)
    rec = app.store.ingest(src)
    duration = rec.probe.duration_seconds
    stages = topo_order([s for t in a.tasks for s in TASK_TO_STAGES[t]])
    times: dict[str, list[float]] = {s: [] for s in stages}
    with PeakRSS() as mem:
        for _ in range(a.repeat):
            app.cache.clear(rec.audio_id)
            for st in stages:  # one stage per call: its dependencies are already cached
                t0 = time.perf_counter()
                run_analysis(app, src, [st], {})
                times[st].append(time.perf_counter() - t0)
    report = {"file": a.file, "duration_seconds": duration, "profile": a.profile, "repeats": a.repeat,
              "stages": [{"stage": s, "mean_seconds": round(statistics.mean(v), 4), "min_seconds": round(min(v), 4),
                          "rtf": round(statistics.mean(v) / duration, 4) if duration else None}
                         for s, v in times.items()],
              "peak_rss_mb": round(mem.peak / 2 ** 20, 1)}
    out.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

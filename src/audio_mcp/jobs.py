"""In-process async jobs: bounded worker pool, progress, cooperative cancellation."""
import contextvars
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Literal

from audio_mcp.envelope import AudioMCPError, ErrorCode, error_response

# (cancel_event, progress_callback) for the job running in this thread; the runner picks it up
JOB_CONTEXT: contextvars.ContextVar = contextvars.ContextVar("audio_mcp_job", default=None)

_FINISHED = ("done", "failed", "cancelled")


@dataclass
class JobInfo:
    job_id: str
    tool: str
    status: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    response: dict | None = None

    @property
    def finished(self) -> bool:
        return self.status in _FINISHED


class JobManager:
    def __init__(self, max_concurrent: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_concurrent), thread_name_prefix="audio-mcp-job")
        self._jobs: dict[str, JobInfo] = {}
        self._events: dict[str, threading.Event] = {}
        self._futures: dict = {}
        self._lock = threading.Lock()

    def _set(self, job: JobInfo, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(job, k, v)
            job.updated_at = time.time()

    def submit(self, tool: str, fn: Callable[[threading.Event, Callable[[int, int], None]], dict]) -> str:
        job = JobInfo(job_id=f"job_{uuid.uuid4().hex[:12]}", tool=tool)
        ev = threading.Event()

        def progress(done: int, total: int):
            self._set(job, progress=round(min(done / total, 0.99), 3) if total else 0.0)

        def run():
            if ev.is_set():
                self._set(job, status="cancelled")
                return
            self._set(job, status="running")
            token = JOB_CONTEXT.set((ev, progress))
            try:
                resp = fn(ev, progress)
                cancelled = resp.get("status") == "error" and (resp.get("error") or {}).get("code") == "CANCELLED"
                self._set(job, status="cancelled" if cancelled else "done", progress=1.0, response=resp)
            except AudioMCPError as e:
                self._set(job, status="cancelled" if e.code == ErrorCode.CANCELLED else "failed",
                          response=error_response(e))
            except Exception as e:  # never leave a job hanging in "running"
                self._set(job, status="failed",
                          response=error_response(AudioMCPError(ErrorCode.STAGE_FAILED, type(e).__name__)))
            finally:
                JOB_CONTEXT.reset(token)

        with self._lock:
            self._jobs[job.job_id], self._events[job.job_id] = job, ev
            self._futures[job.job_id] = self._pool.submit(run)
        return job.job_id

    def get(self, job_id: str) -> JobInfo:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise AudioMCPError(ErrorCode.JOB_NOT_FOUND, f"unknown job {job_id}")
        return job

    def cancel(self, job_id: str) -> JobInfo:
        job = self.get(job_id)
        self._events[job_id].set()
        if self._futures[job_id].cancel():  # still queued: never starts
            self._set(job, status="cancelled")
        return job

    def prune(self, older_than: float) -> int:
        with self._lock:
            old = [j for j, info in self._jobs.items() if info.finished and info.updated_at < older_than]
            for j in old:
                del self._jobs[j], self._events[j], self._futures[j]
        return len(old)

    def shutdown(self):
        for ev in self._events.values():
            ev.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

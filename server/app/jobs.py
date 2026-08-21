"""Async job queue with SSE progress.

Renders take minutes. Every mutating endpoint returns a job id immediately and
the client watches /jobs/{id}/events -- the UI must never block, because
"death by a thousand loading screens" is the documented way this class of tool
fails its users.
"""

import asyncio
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    project_id: str | None = None
    status: str = "queued"            # queued | running | done | error
    progress: float = 0.0             # 0..1, meaningful only if determinate
    # ACE-Step reports no progress signal, so a percentage would be invented.
    # Say so, and let the client show elapsed time instead of a fake bar.
    determinate: bool = True
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    queue_position: int = 0
    _subscribers: list = field(default_factory=list, repr=False)

    def public(self):
        return {
            "id": self.id, "kind": self.kind, "project_id": self.project_id,
            "status": self.status, "progress": round(self.progress, 3),
            "determinate": self.determinate,
            "queue_position": self.queue_position,
            "message": self.message, "result": self.result, "error": self.error,
            "elapsed_s": round((self.finished_at or time.time()) - self.created_at, 1),
        }


class JobQueue:
    def __init__(self, concurrency: int = 1):
        self._jobs: dict[str, Job] = {}
        self._concurrency = concurrency
        self._sem: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop):
        """Called once at startup.

        Sync route handlers run in FastAPI's threadpool, where there is no
        running loop, so submit() cannot rely on create_task. Holding the loop
        lets any thread schedule work onto it.
        """
        self._loop = loop
        self._sem = asyncio.Semaphore(self._concurrency)

    MAX_FINISHED = 200

    def _evict(self):
        """Finished jobs accumulated forever; keep the most recent."""
        finished = [j for j in self._jobs.values()
                    if j.status in ("done", "error") and not j._subscribers]
        excess = len(finished) - self.MAX_FINISHED
        if excess > 0:
            for j in sorted(finished, key=lambda j: j.finished_at or 0)[:excess]:
                self._jobs.pop(j.id, None)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, project_id: str | None = None) -> list[dict]:
        jobs = self._jobs.values()
        if project_id:
            jobs = [j for j in jobs if j.project_id == project_id]
        return [j.public() for j in sorted(jobs, key=lambda j: j.created_at, reverse=True)]

    def _renumber(self):
        """Position among jobs still waiting. One GPU means a real queue."""
        waiting = [j for j in self._jobs.values() if j.status == "queued"]
        for i, j in enumerate(sorted(waiting, key=lambda j: j.created_at), start=1):
            j.queue_position = i
        for j in self._jobs.values():
            if j.status != "queued":
                j.queue_position = 0

    def submit(self, kind: str, fn: Callable, project_id: str | None = None) -> Job:
        """fn(report) -> result, run in a worker thread. report(progress, msg)."""
        job = Job(id=f"job_{uuid.uuid4().hex[:10]}", kind=kind, project_id=project_id)
        self._jobs[job.id] = job
        self._evict()
        self._renumber()
        coro = self._run(job, fn)
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            if self._loop is None:
                coro.close()
                raise RuntimeError("job queue is not bound to an event loop")
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        return job

    async def _run(self, job: Job, fn: Callable):
        loop = asyncio.get_running_loop()

        def report(progress: float, message: str = ""):
            # A negative progress means "running, but I cannot know how far".
            if progress < 0:
                job.determinate = False
            else:
                job.progress = max(0.0, min(1.0, progress))
            if message:
                job.message = message
            loop.call_soon_threadsafe(self._notify, job)

        self._loop = self._loop or loop

        if self._sem is None:
            self._sem = asyncio.Semaphore(self._concurrency)
        async with self._sem:
            job.started_at = time.time()
            job.status = "running"
            self._notify(job)
            try:
                job.result = await asyncio.to_thread(fn, report)
                job.status = "done"
                job.progress = 1.0
                job.message = job.message or "complete"
            except Exception as exc:                       # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "failed"
                traceback.print_exc()
            finally:
                job.finished_at = time.time()
                self._renumber()
                self._notify(job)

    def _notify(self, job: Job):
        for q in list(job._subscribers):
            try:
                q.put_nowait(job.public())
            except asyncio.QueueFull:
                pass

    async def events(self, job: Job):
        """Async generator of SSE payloads for one job."""
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        job._subscribers.append(q)
        try:
            yield job.public()                 # current state first
            while job.status in ("queued", "running"):
                try:
                    yield await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"heartbeat": True, "id": job.id}
            # Drain anything already queued so the terminal state is not missed.
            while not q.empty():
                yield q.get_nowait()
            yield job.public()
        finally:
            if q in job._subscribers:
                job._subscribers.remove(q)


queue = JobQueue()

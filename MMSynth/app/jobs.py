"""An async job queue with SSE progress.

Renders take seconds to minutes. Every route that starts one returns a job id
immediately and the client watches the stream, because a UI that blocks on a GPU
is a UI that feels broken at six seconds and abandoned at sixty.

One GPU means one worker. `concurrency=1` is not a placeholder -- it is the
truth about the machine, and a queue position is more honest than four requests
fighting over the same card.
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class JobCancelled(Exception):
    """Raised inside a worker when the client asked it to stop.

    `asyncio.to_thread` cannot be cancelled from outside, so cancellation has to
    be cooperative. Every long step already calls report() to say where it is,
    which makes report() the natural place to notice and bail out.
    """


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"          # queued | running | done | error | cancelled
    cancel_requested: bool = False
    progress: float = 0.0
    # Some models report nothing at all. Say so rather than inventing a bar.
    determinate: bool = True
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    queue_position: int = 0
    _subscribers: list = field(default_factory=list, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "determinate": self.determinate,
            "queue_position": self.queue_position,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "elapsed_s": round((self.finished_at or time.time()) - self.created_at, 1),
        }


class JobQueue:
    def __init__(self, concurrency: int = 1):
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._concurrency = concurrency
        self._sem: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once at startup.

        Sync route handlers run in the threadpool, where there is no running
        loop, so submit() cannot rely on create_task. Holding the loop here is
        what lets a handler schedule work from a thread.
        """
        self._loop = loop
        self._sem = asyncio.Semaphore(self._concurrency)

    # -- reading -----------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _waiting(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "queued")

    # -- writing -----------------------------------------------------------

    def submit(self, kind: str, fn: Callable[[Callable], Any],
               idempotency_key: str | None = None) -> Job:
        """Queue `fn(report)`.

        An idempotency key returns the job the first request made rather than
        starting a second one. A lost response is common and a duplicated GPU
        render is expensive, so the client retrying must be free.
        """
        if idempotency_key:
            existing = self._by_key.get(idempotency_key)
            if existing and existing in self._jobs:
                return self._jobs[existing]

        job = Job(id=uuid.uuid4().hex, kind=kind)
        job.queue_position = self._waiting()
        self._jobs[job.id] = job
        if idempotency_key:
            self._by_key[idempotency_key] = job.id

        if self._loop is None:                                # pragma: no cover
            raise RuntimeError("job queue was never bound to a loop")
        asyncio.run_coroutine_threadsafe(self._run(job, fn), self._loop)
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("done", "error", "cancelled"):
            return True                # already stopped; asking again is harmless
        job.cancel_requested = True
        if job.status == "queued":
            self._finish(job, "cancelled", message="cancelled before it started")
        return True

    # -- internals ---------------------------------------------------------

    async def _run(self, job: Job, fn: Callable[[Callable], Any]) -> None:
        assert self._sem is not None
        async with self._sem:
            if job.cancel_requested or job.status == "cancelled":
                return
            job.status = "running"
            job.started_at = time.time()
            job.queue_position = 0
            self._publish(job)

            def report(progress: float, message: str = "") -> None:
                if job.cancel_requested:
                    raise JobCancelled()
                if progress < 0:
                    job.determinate = False
                else:
                    job.determinate = True
                    job.progress = max(0.0, min(1.0, progress))
                if message:
                    job.message = message
                self._publish(job)

            try:
                result = await asyncio.to_thread(fn, report)
            except JobCancelled:
                self._finish(job, "cancelled", message="cancelled")
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                self._finish(job, "error", error=f"{type(exc).__name__}: {exc}")
            else:
                job.result = result
                self._finish(job, "done", progress=1.0)

    def _finish(self, job: Job, status: str, progress: float | None = None,
                message: str = "", error: str | None = None) -> None:
        job.status = status
        job.finished_at = time.time()
        if progress is not None:
            job.progress = progress
        if message:
            job.message = message
        if error:
            job.error = error
        self._publish(job)

    def _publish(self, job: Job) -> None:
        for queue in list(job._subscribers):
            try:
                queue.put_nowait(job.public())
            except asyncio.QueueFull:                         # pragma: no cover
                pass

    def subscribe(self, job: Job) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        job._subscribers.append(queue)
        queue.put_nowait(job.public())
        return queue

    def unsubscribe(self, job: Job, queue: asyncio.Queue) -> None:
        if queue in job._subscribers:
            job._subscribers.remove(queue)


queue = JobQueue(concurrency=1)

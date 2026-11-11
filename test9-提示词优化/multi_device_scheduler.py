"""Thread-safe fair scheduling primitives for a single Test6 server.

The public gateway gives every hardware token a stable opaque principal id.
This module uses that id only for scheduling: one noisy device cannot occupy
all workers while jobs from other devices are waiting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Generic, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ScheduledItem(Generic[T]):
    value: T
    device_id: str
    enqueued_at: float


@dataclass(frozen=True)
class SubmitResult:
    accepted: bool
    reason: str | None
    pending_total: int
    pending_for_device: int


class FairDeviceQueue(Generic[T]):
    """Bounded round-robin queue with a per-device active-job limit.

    ``capacity=0`` means that the in-memory job-reference queue is unbounded.
    Uploaded image bytes are not stored here; the pipeline persists those to
    the job directory before enqueueing.
    """

    def __init__(
        self,
        *,
        capacity: int,
        per_device_pending: int,
        per_device_active: int = 1,
    ) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if per_device_pending < 1:
            raise ValueError("per_device_pending must be positive")
        if per_device_active < 1:
            raise ValueError("per_device_active must be positive")
        self.capacity = capacity
        self.per_device_pending = per_device_pending
        self.per_device_active = per_device_active
        self._condition = threading.Condition()
        self._queues: dict[str, deque[ScheduledItem[T]]] = {}
        self._rotation: deque[str] = deque()
        self._active_by_device: dict[str, int] = {}
        self._pending_total = 0
        self._closed = False

    def submit(self, value: T, device_id: str) -> SubmitResult:
        normalized = str(device_id or "direct")[:128]
        with self._condition:
            if self._closed:
                return SubmitResult(False, "closed", self._pending_total, 0)
            device_queue = self._queues.get(normalized)
            device_pending = len(device_queue) if device_queue is not None else 0
            if self.capacity and self._pending_total >= self.capacity:
                return SubmitResult(
                    False, "queue_full", self._pending_total, device_pending
                )
            if device_pending >= self.per_device_pending:
                return SubmitResult(
                    False,
                    "device_queue_full",
                    self._pending_total,
                    device_pending,
                )
            if device_queue is None:
                device_queue = deque()
                self._queues[normalized] = device_queue
                self._rotation.append(normalized)
            device_queue.append(
                ScheduledItem(
                    value=value,
                    device_id=normalized,
                    enqueued_at=time.monotonic(),
                )
            )
            self._pending_total += 1
            self._condition.notify_all()
            return SubmitResult(
                True,
                None,
                self._pending_total,
                len(device_queue),
            )

    def get(self, timeout: float | None = None) -> ScheduledItem[T] | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                rotations = len(self._rotation)
                for _ in range(rotations):
                    device_id = self._rotation.popleft()
                    device_queue = self._queues.get(device_id)
                    if not device_queue:
                        self._queues.pop(device_id, None)
                        continue
                    active = self._active_by_device.get(device_id, 0)
                    if active >= self.per_device_active:
                        self._rotation.append(device_id)
                        continue
                    item = device_queue.popleft()
                    self._pending_total -= 1
                    self._active_by_device[device_id] = active + 1
                    if device_queue:
                        self._rotation.append(device_id)
                    else:
                        self._queues.pop(device_id, None)
                    return item

                if self._closed:
                    return None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def task_done(self, item: ScheduledItem[T]) -> None:
        with self._condition:
            active = self._active_by_device.get(item.device_id, 0)
            if active <= 0:
                raise ValueError("task_done called without an active item")
            if active == 1:
                self._active_by_device.pop(item.device_id, None)
            else:
                self._active_by_device[item.device_id] = active - 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def reopen(self) -> None:
        with self._condition:
            self._closed = False
            self._condition.notify_all()

    def qsize(self) -> int:
        with self._condition:
            return self._pending_total

    def active_count(self) -> int:
        with self._condition:
            return sum(self._active_by_device.values())

    def active_device_count(self) -> int:
        with self._condition:
            return len(self._active_by_device)

    def pending_device_count(self) -> int:
        with self._condition:
            return len(self._queues)

    def snapshot(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "capacity": self.capacity,
                "pending_jobs": self._pending_total,
                "pending_devices": len(self._queues),
                "active_jobs": sum(self._active_by_device.values()),
                "active_devices": len(self._active_by_device),
                "per_device_pending": self.per_device_pending,
                "per_device_active": self.per_device_active,
                "closed": self._closed,
            }


class ConcurrentJobWorker(Generic[T]):
    """Run one dedicated handler per consumer over a fair device queue."""

    def __init__(
        self,
        handlers: Sequence[Callable[[T], object]],
        *,
        capacity: int,
        per_device_pending: int,
        per_device_active: int = 1,
        thread_name_prefix: str = "test6-worker",
        on_error: Callable[[T, Exception], None] | None = None,
    ) -> None:
        if not handlers:
            raise ValueError("at least one handler is required")
        self.handlers = tuple(handlers)
        self.pending = FairDeviceQueue[T](
            capacity=capacity,
            per_device_pending=per_device_pending,
            per_device_active=per_device_active,
        )
        self.thread_name_prefix = thread_name_prefix
        self.on_error = on_error
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self._stats_lock = threading.Lock()
        self._completed = 0
        self._failed = 0

    @property
    def worker_count(self) -> int:
        return len(self.handlers)

    def start(self) -> None:
        if self.threads and any(thread.is_alive() for thread in self.threads):
            return
        self.stop_event.clear()
        self.pending.reopen()
        self.threads = []
        for index, handler in enumerate(self.handlers, 1):
            thread = threading.Thread(
                target=self._loop,
                args=(handler,),
                name=f"{self.thread_name_prefix}-{index}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def stop(self, join_timeout: float = 3.0) -> None:
        self.stop_event.set()
        self.pending.close()
        deadline = time.monotonic() + max(0.0, join_timeout)
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def submit(self, value: T, device_id: str) -> SubmitResult:
        return self.pending.submit(value, device_id)

    def alive_count(self) -> int:
        return sum(thread.is_alive() for thread in self.threads)

    def is_alive(self) -> bool:
        return self.alive_count() > 0

    def is_healthy(self) -> bool:
        return self.alive_count() == self.worker_count

    def snapshot(self) -> dict[str, int | bool]:
        snapshot = self.pending.snapshot()
        with self._stats_lock:
            snapshot.update(
                {
                    "configured_workers": self.worker_count,
                    "alive_workers": self.alive_count(),
                    "completed_jobs": self._completed,
                    "failed_jobs": self._failed,
                }
            )
        return snapshot

    def _loop(self, handler: Callable[[T], object]) -> None:
        while not self.stop_event.is_set():
            item = self.pending.get(timeout=0.5)
            if item is None:
                continue
            succeeded = False
            try:
                handler(item.value)
                succeeded = True
            except Exception as exc:  # keep later device jobs serviceable
                if self.on_error is not None:
                    try:
                        self.on_error(item.value, exc)
                    except Exception:
                        pass
            finally:
                with self._stats_lock:
                    if succeeded:
                        self._completed += 1
                    else:
                        self._failed += 1
                self.pending.task_done(item)


__all__ = [
    "ConcurrentJobWorker",
    "FairDeviceQueue",
    "ScheduledItem",
    "SubmitResult",
]

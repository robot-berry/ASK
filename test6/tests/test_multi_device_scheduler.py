from __future__ import annotations

import threading
import time
import unittest

from multi_device_scheduler import ConcurrentJobWorker, FairDeviceQueue


class FairDeviceQueueTests(unittest.TestCase):
    def test_round_robin_and_one_active_job_per_device(self) -> None:
        pending = FairDeviceQueue[str](
            capacity=20,
            per_device_pending=4,
            per_device_active=1,
        )
        self.assertTrue(pending.submit("a1", "device-a").accepted)
        self.assertTrue(pending.submit("a2", "device-a").accepted)
        self.assertTrue(pending.submit("b1", "device-b").accepted)

        first = pending.get(timeout=0)
        second = pending.get(timeout=0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual((first.value, second.value), ("a1", "b1"))
        self.assertIsNone(pending.get(timeout=0))

        pending.task_done(first)
        third = pending.get(timeout=0)
        self.assertIsNotNone(third)
        assert third is not None
        self.assertEqual(third.value, "a2")
        pending.task_done(second)
        pending.task_done(third)

    def test_total_and_per_device_limits_are_distinct(self) -> None:
        pending = FairDeviceQueue[str](
            capacity=3,
            per_device_pending=2,
        )
        self.assertTrue(pending.submit("a1", "a").accepted)
        self.assertTrue(pending.submit("a2", "a").accepted)
        per_device_rejection = pending.submit("a3", "a")
        self.assertFalse(per_device_rejection.accepted)
        self.assertEqual(per_device_rejection.reason, "device_queue_full")
        self.assertTrue(pending.submit("b1", "b").accepted)
        total_rejection = pending.submit("c1", "c")
        self.assertFalse(total_rejection.accepted)
        self.assertEqual(total_rejection.reason, "queue_full")

    def test_zero_capacity_means_unbounded_total_queue(self) -> None:
        pending = FairDeviceQueue[int](capacity=0, per_device_pending=10)
        for index in range(10):
            self.assertTrue(pending.submit(index, f"device-{index}").accepted)
        self.assertEqual(pending.qsize(), 10)


class ConcurrentJobWorkerTests(unittest.TestCase):
    def test_distinct_devices_run_concurrently_while_same_device_waits(self) -> None:
        release = threading.Event()
        started: list[str] = []
        started_lock = threading.Lock()

        def handler(value: str) -> None:
            with started_lock:
                started.append(value)
            release.wait(2)

        worker = ConcurrentJobWorker(
            [handler, handler],
            capacity=20,
            per_device_pending=4,
            per_device_active=1,
        )
        worker.submit("a1", "device-a")
        worker.submit("a2", "device-a")
        worker.submit("b1", "device-b")
        worker.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with started_lock:
                if len(started) >= 2:
                    break
            time.sleep(0.01)
        with started_lock:
            self.assertCountEqual(started[:2], ["a1", "b1"])
            self.assertNotIn("a2", started[:2])
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with started_lock:
                if "a2" in started:
                    break
            time.sleep(0.01)
        worker.stop()
        with started_lock:
            self.assertIn("a2", started)
        snapshot = worker.snapshot()
        self.assertEqual(snapshot["completed_jobs"], 3)
        self.assertEqual(snapshot["failed_jobs"], 0)

    def test_handler_failure_releases_device_and_later_job_runs(self) -> None:
        completed = threading.Event()
        errors: list[tuple[str, str]] = []

        def handler(value: str) -> None:
            if value == "bad":
                raise RuntimeError("simulated")
            completed.set()

        worker = ConcurrentJobWorker(
            [handler],
            capacity=10,
            per_device_pending=4,
            per_device_active=1,
            on_error=lambda value, exc: errors.append((value, str(exc))),
        )
        worker.submit("bad", "device-a")
        worker.submit("good", "device-a")
        worker.start()
        self.assertTrue(completed.wait(2))
        worker.stop()

        self.assertEqual(errors, [("bad", "simulated")])
        snapshot = worker.snapshot()
        self.assertEqual(snapshot["completed_jobs"], 1)
        self.assertEqual(snapshot["failed_jobs"], 1)
        self.assertEqual(snapshot["active_jobs"], 0)

    def test_clean_stop_can_be_restarted(self) -> None:
        values: list[str] = []
        completed = threading.Event()

        def handler(value: str) -> None:
            values.append(value)
            completed.set()

        worker = ConcurrentJobWorker(
            [handler],
            capacity=10,
            per_device_pending=4,
        )
        worker.start()
        worker.stop()
        completed.clear()
        worker.start()
        self.assertTrue(worker.submit("after-restart", "device-b").accepted)
        self.assertTrue(completed.wait(2))
        worker.stop()

        self.assertEqual(values, ["after-restart"])
        self.assertEqual(worker.snapshot()["completed_jobs"], 1)


if __name__ == "__main__":
    unittest.main()

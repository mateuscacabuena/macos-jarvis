"""Enqueue a Jarvis command on the Redis queue and follow it to completion.

Usage:
    uv run python test_async.py "what time is it?"

Requires the worker to be running separately:
    uv run saq jarvis.queue.worker.settings
"""

import asyncio
import sys
import time

from saq import Queue
from saq.job import Status

from jarvis.config import Settings


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "What time is it?"
    settings = Settings()  # type: ignore[call-arg]

    queue = Queue.from_url(settings.redis_url, name="jarvis")
    await queue.connect()

    t0 = time.monotonic()
    job = await queue.enqueue("process_jarvis_command", user_prompt=prompt)
    if job is None:
        print("[test_async] Job was not enqueued (duplicate?)")
        await queue.disconnect()
        return

    elapsed_ms = (time.monotonic() - t0) * 1000
    print(f"[test_async] Enqueued job {job.id} in {elapsed_ms:.1f}ms")

    while job.status not in (Status.COMPLETE, Status.FAILED, Status.ABORTED):
        print(f"[test_async] status={job.status.value}")
        # refresh(1) blocks up to 1s waiting for a pubsub update, raising
        # TimeoutError if the job is still running — that's expected, not a failure.
        try:
            await job.refresh(1)
        except TimeoutError:
            continue

    print(f"[test_async] Final status: {job.status.value} ({time.monotonic() - t0:.1f}s total)")
    if job.status == Status.COMPLETE:
        print(f"[test_async] Result: {job.result}")
    else:
        print(f"[test_async] Error: {job.error}")

    await queue.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

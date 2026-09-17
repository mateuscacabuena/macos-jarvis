from saq import Queue
from saq.job import Job

from jarvis.config import Settings

_queue: Queue | None = None


def get_queue(settings: Settings) -> Queue:
    """Return the shared queue connection, creating it on first use."""
    global _queue
    if _queue is None:
        _queue = Queue.from_url(settings.redis_url, name="jarvis")
    return _queue


async def enqueue_command(prompt: str, settings: Settings) -> str:
    """Queue a text command for the background worker and return its job id."""
    queue = get_queue(settings)
    await queue.connect()
    job = await queue.enqueue("process_jarvis_command", user_prompt=prompt)
    if job is None:
        raise RuntimeError("Job was not enqueued")
    return job.key


async def get_job(job_id: str, settings: Settings) -> Job | None:
    queue = get_queue(settings)
    await queue.connect()
    return await queue.job(job_id)

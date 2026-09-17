from saq import Queue

from jarvis.config import Settings
from jarvis.queue.tasks import process_jarvis_command, startup

_settings = Settings()  # type: ignore[call-arg]

queue = Queue.from_url(_settings.redis_url, name="jarvis")

settings = {
    "queue": queue,
    "functions": [process_jarvis_command],
    "startup": startup,
    "concurrency": 1,
}

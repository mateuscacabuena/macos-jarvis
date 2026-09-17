"""HTTP entry point for queuing Jarvis commands from outside the voice pipeline.

Run with:
    uv run uvicorn jarvis.queue.api:app --port 8008

Requires the worker to be running separately:
    uv run saq jarvis.queue.worker.settings
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jarvis.config import Settings
from jarvis.queue.producer import enqueue_command, get_job

app = FastAPI(title="Jarvis Task Queue API")
_settings = Settings()  # type: ignore[call-arg]


class TaskRequest(BaseModel):
    prompt: str


class TaskResponse(BaseModel):
    job_id: str


class TaskStatus(BaseModel):
    job_id: str
    status: str
    result: str | None = None
    error: str | None = None


@app.post("/tasks", response_model=TaskResponse)
async def create_task(req: TaskRequest) -> TaskResponse:
    job_id = await enqueue_command(req.prompt, _settings)
    return TaskResponse(job_id=job_id)


@app.get("/tasks/{job_id}", response_model=TaskStatus)
async def read_task(job_id: str) -> TaskStatus:
    job = await get_job(job_id, _settings)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return TaskStatus(job_id=job_id, status=job.status.value, result=job.result, error=job.error)

# Jarvis (macos-jarvis)

Voice-controlled macOS assistant for Apple Silicon. Pipeline: wake word → record →
transcribe → (optional camera) → think (LLM + tools) → act → speak.

## Stack

- Python 3.11+, managed with `uv`
- Wake word: openWakeWord (`hey_jarvis`)
- STT: Whisper Small (MLX)
- TTS: Kokoro 82M (MLX)
- Vision: OpenCV + Continuity Camera
- Reasoning/tool routing: Groq API (Llama models)
- Config: `pydantic-settings` (`src/jarvis/config.py`)
- Tracing: Langfuse (instrumented in `src/jarvis/brain.py`)

## Layout

```
src/jarvis/
├── main.py      # Pipeline orchestration
├── wake.py      # Wake word detection
├── audio.py     # Microphone recording with silence detection
├── ears.py      # Speech-to-text
├── eyes.py      # Camera capture
├── brain.py     # LLM API + tool execution loop
├── hands.py     # Apple Shortcuts, file ops, Spotlight search
├── mouth.py     # Text-to-speech
└── config.py    # Settings
```

Fully async pipeline; wake word detection runs on a background thread, heavy model
inference is offloaded to thread executors.

## Project provenance

This repository is a personal fork, created for study purposes, of the upstream
project https://github.com/luccaparadeda/macos-jarvis. The fork lives at
https://github.com/mateuscacabuena/macos-jarvis (git remote `fork`).

All contributions made here are for learning/experimentation and to extend/improve
that fork. They are not official upstream contributions unless explicitly submitted
back via a PR to the original project.

## Git workflow

- Always merge and push finished work to the `main` branch of the fork
  (`git push fork main`), not to the upstream `origin` remote.
- Do not leave stale feature/dependabot branches lingering on the fork after
  their work has been merged into `main` — delete them from the remote once
  merged.

from jarvis.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    settings = Settings()
    assert settings.groq_api_key == "test-key-123"
    assert settings.groq_model == "openai/gpt-oss-120b"


def test_settings_defaults():
    settings = Settings(groq_api_key="k")
    assert settings.whisper_model == "mlx-community/whisper-small-mlx"
    assert settings.kokoro_model == "mlx-community/Kokoro-82M-bf16"
    assert settings.wake_model == "hey_jarvis"
    assert settings.camera_index == 0
    assert settings.silence_threshold == 0.0
    assert settings.silence_multiplier == 3.0
    assert settings.silence_duration == 1.5
    assert "look" in settings.vision_keywords
    assert "see" in settings.vision_keywords


def test_settings_override(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    settings = Settings()
    assert settings.groq_model == "llama-3.1-8b-instant"
    assert settings.camera_index == 2

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    groq_vision_model: str = "qwen/qwen3.6-27b"
    whisper_model: str = "mlx-community/whisper-small-mlx"
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.6
    camera_index: int = 0
    silence_threshold: float = 0.0
    silence_multiplier: float = 3.0
    silence_duration: float = 1.5
    silence_floor: float = 0.01
    silence_ceiling: float = 0.08
    no_voice_timeout: float = 5.0
    vision_keywords: list[str] = [
        "look",
        "see",
        "show",
        "what is",
        "camera",
        "screen",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram MTProto credentials
    telegram_api_id: int
    telegram_api_hash: str
    telegram_session_name: str = "booking_agent"

    # Accounts & targets
    owner_telegram: str = "@boris_nekit"
    owner_escalation_chat: str = "me"
    target_chat: str = "@vlad_dev_7"
    self_test_mode: bool = False

    # OpenRouter credentials & models
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-haiku-5.5"
    openrouter_stt_model: str = "google/gemini-3.6-flash"

    # Guest & booking baseline facts
    guest_name: str = "Nikita"
    party_size: int = 2
    booking_time: str = "17:00"
    restaurant_city: str = "London"
    restaurant_timezone: str = "Europe/London"

    # Voice & runtime settings
    tts_voice: str = "en-GB-RyanNeural"
    debounce_seconds: float = 6.0
    decision_log_path: Path = Path("logs/decision.log")

    def _london_now(self, now: datetime | None = None) -> datetime:
        tz = ZoneInfo(self.restaurant_timezone)
        if now is None:
            return datetime.now(tz)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(tz)

    def london_today_iso(self, now: datetime | None = None) -> str:
        return self._london_now(now).strftime("%Y-%m-%d")

    def london_today_human(self, now: datetime | None = None) -> str:
        dt = self._london_now(now)
        return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B %Y')}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

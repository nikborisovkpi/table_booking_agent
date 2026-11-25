import asyncio
import sys

from typing import Any

from telethon import TelegramClient

from src.config import Settings, get_settings
from src.decision_logger import DecisionLogger
from src.human_escalation import TelegramOwnerBridge
from src.llm_agent import BookingAgent
from src.telegram_client import TelegramBookingRunner, resolve_target_entity
from src.voice import VoiceService

_REAL_MANAGER_HANDLES = {"@vlad_dev_7", "vlad_dev_7", "https://t.me/vlad_dev_7"}


def configure_console_utf8(*streams: Any) -> None:
    targets = streams or (sys.stdout, sys.stderr)
    for stream in targets:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _validate_runtime_settings(settings: Settings) -> None:
    key = settings.openrouter_api_key.strip()
    if not key or key in {
        "your_openrouter_api_key_here",
        "your_google_api_key_here",
        "your_gemini_api_key_here",
    }:
        raise SystemExit(
            "❌ Помилка: вкажіть реальний API-ключ (GEMINI_API_KEY або OPENROUTER_API_KEY) у файлі .env перед запуском агента."
        )
    target_norm = settings.target_chat.strip().lower()
    if settings.self_test_mode:
        if target_norm == settings.owner_escalation_chat.strip().lower():
            raise SystemExit(
                "❌ Помилка конфігурації: у режимі SELF_TEST_MODE=true TARGET_CHAT не може збігатися з "
                "OWNER_ESCALATION_CHAT ('me'). Створіть окрему приватну групу на себе (наприклад, "
                "'Test London Restaurant') і вкажіть її назву в TARGET_CHAT."
            )
        if target_norm in _REAL_MANAGER_HANDLES:
            raise SystemExit(
                "❌ Запобіжник: увімкнено SELF_TEST_MODE=true, але TARGET_CHAT вказує на реального менеджера "
                "@vlad_dev_7! Для самотестування вкажіть назву вашої тестової групи (наприклад, "
                "TARGET_CHAT=Test London Restaurant), а для реальної броні встановіть SELF_TEST_MODE=false."
            )


async def run_agent() -> None:
    settings = get_settings()
    _validate_runtime_settings(settings)

    mode_label = "SELF-TEST (один акаунт у тестовій групі)" if settings.self_test_mode else "LIVE PRODUCTION"
    print(
        "========================================\n"
        "🍽️  London Restaurant Booking Agent\n"
        f"Mode             : {mode_label}\n"
        f"Target chat      : {settings.target_chat}\n"
        f"Owner Q&A chat   : {settings.owner_escalation_chat} ({settings.owner_telegram})\n"
        f"Guest / Party    : {settings.guest_name} / {settings.party_size} guests\n"
        f"Requested slot   : {settings.booking_time} today ({settings.london_today_human()})\n"
        f"LLM Endpoint     : {settings.openrouter_base_url}\n"
        f"Main LLM         : {settings.openrouter_model}\n"
        f"Voice STT / TTS  : {settings.openrouter_stt_model} / {settings.tts_voice}\n"
        f"Decision log     : {settings.decision_log_path}\n"
        "========================================\n",
        flush=True,
    )

    client = TelegramClient(
        settings.telegram_session_name,
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )

    await client.start()
    runner: TelegramBookingRunner | None = None
    try:
        target_entity = await resolve_target_entity(client, settings.target_chat)

        voice_service = VoiceService(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            stt_model=settings.openrouter_stt_model,
            tts_voice=settings.tts_voice,
        )

        owner_bridge = TelegramOwnerBridge(
            client=client,
            escalation_chat=settings.owner_escalation_chat,
        )
        owner_bridge.register_telethon_handler(voice_service=voice_service)

        agent = BookingAgent(
            settings=settings,
            ask_owner_fn=owner_bridge.ask_owner,
        )
        decision_logger = DecisionLogger(settings.decision_log_path)

        runner = TelegramBookingRunner(
            settings=settings,
            client=client,
            target_entity=target_entity,
            agent=agent,
            voice_service=voice_service,
            decision_logger=decision_logger,
        )
        runner.register_telethon_handlers()

        await runner.initiate_opening_turn()
        print("⏳ Очікую відповіді менеджера... (Натисніть Ctrl+C для ручної зупинки)\n", flush=True)
        await runner.completed_event.wait()
        # Brief pause so final Telegram network packets flush cleanly
        await asyncio.sleep(1.0)
    finally:
        if runner is not None:
            await runner.close()
        await client.disconnect()


def main() -> None:
    configure_console_utf8()
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\n⏹️  Роботу агента зупинено користувачем.", file=sys.stderr)


if __name__ == "__main__":
    main()

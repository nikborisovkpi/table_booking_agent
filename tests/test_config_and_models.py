from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import Settings
from src.models import AgentTurnOutput, ConfirmedBookingDetails, MessageModality


def test_settings_loads_defaults_and_computes_london_date():
    settings = Settings(
        _env_file=None,
        telegram_api_id=31939235,
        telegram_api_hash="test_hash",
        openrouter_api_key="sk-or-test",
    )
    assert settings.guest_name == "Nikita"
    assert settings.party_size == 2
    assert settings.booking_time == "17:00"
    assert settings.openrouter_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.openrouter_model == "gemini-3.8-flash"
    assert settings.openrouter_stt_model == "gemini-3.8-flash"
    assert settings.owner_escalation_chat == "me"
    assert settings.decision_log_path == Path("logs/decision.log")

    # 2026-07-15 23:30 UTC is 2026-07-16 00:30 BST in Europe/London
    sample_utc = datetime(2026, 7, 15, 23, 30, tzinfo=timezone.utc)
    assert settings.london_today_iso(sample_utc) == "2026-07-16"
    assert settings.london_today_human(sample_utc) == "Thursday, 16 July 2026"


def test_agent_turn_output_requires_confirmed_details_when_booking_confirmed():
    with pytest.raises(ValidationError, match="confirmed_details"):
        AgentTurnOutput(
            understood="Manager confirmed the table",
            decision="Send final summary",
            reply_text="Thank you! Confirming for two at 17:00 today.",
            booking_confirmed=True,
            confirmed_details=None,
        )


def test_confirmed_details_rejects_unknown_or_blank_restaurant_name():
    with pytest.raises(ValidationError, match="restaurant_name"):
        ConfirmedBookingDetails(
            time="17:00",
            date="2026-11-25",
            party_size=2,
            restaurant_name="unknown",
        )


def test_valid_confirmed_agent_turn_output():
    turn = AgentTurnOutput(
        understood="Manager confirmed 17:00 today at The Ivy Soho Brasserie",
        decision="Repeat all four confirmed booking details politely and finish",
        reply_text=(
            "Lovely, thank you ever so much! Just to confirm: a table for 2 under Nikita "
            "at 17:00 today, Wednesday 25 November 2026, at The Ivy Soho Brasserie."
        ),
        booking_confirmed=True,
        confirmed_details=ConfirmedBookingDetails(
            time="17:00",
            date="Wednesday, 25 November 2026",
            party_size=2,
            restaurant_name="The Ivy Soho Brasserie",
        ),
    )
    assert turn.booking_confirmed is True
    assert turn.confirmed_details is not None
    assert turn.confirmed_details.restaurant_name == "The Ivy Soho Brasserie"
    assert MessageModality.VOICE.value == "VOICE"


def test_settings_accepts_gemini_api_key_alias(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test_key")
    settings = Settings(
        _env_file=None,
        telegram_api_id=31939235,
        telegram_api_hash="test_hash",
    )
    assert settings.openrouter_api_key == "AIza_test_key"

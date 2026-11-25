import io
import pytest

from src.config import Settings
from src.main import _validate_runtime_settings, configure_console_utf8


def test_configure_console_utf8_prevents_cp1251_unicode_encode_error():
    raw_buf = io.BytesIO()
    cp1251_stream = io.TextIOWrapper(raw_buf, encoding="cp1251", errors="strict")

    # Confirm cp1251 fails before reconfiguration
    with pytest.raises(UnicodeEncodeError):
        cp1251_stream.write("🍽️ Привіт із Києва!\n")

    configure_console_utf8(cp1251_stream)
    cp1251_stream.write("🍽️ Привіт із Києва!\n")
    cp1251_stream.flush()

    assert "🍽️ Привіт із Києва!" in raw_buf.getvalue().decode("utf-8")


def test_validate_runtime_settings_blocks_real_manager_when_self_test_mode_enabled():
    settings = Settings(
        _env_file=None,
        telegram_api_id=31939235,
        telegram_api_hash="test_hash",
        openrouter_api_key="sk-or-valid-key",
        self_test_mode=True,
        target_chat="@vlad_dev_7",
    )
    with pytest.raises(SystemExit, match="vlad_dev_7"):
        _validate_runtime_settings(settings)

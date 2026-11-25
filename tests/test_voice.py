import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.models import MessageModality
from src.voice import VoiceService, resolve_reply_modality


async def _generate_sample_audio(target_path: Path, codec_args: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=0.2",
        *codec_args,
        str(target_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode == 0, stderr.decode("utf-8", errors="replace")


def test_resolve_reply_modality_mirrors_incoming():
    assert resolve_reply_modality(None) == MessageModality.TEXT
    assert resolve_reply_modality(MessageModality.TEXT) == MessageModality.TEXT
    assert resolve_reply_modality(MessageModality.VOICE) == MessageModality.VOICE


async def test_transcribe_voice_converts_ogg_to_wav_and_calls_openrouter_stt(tmp_path: Path):
    sample_ogg = tmp_path / "incoming.ogg"
    await _generate_sample_audio(sample_ogg, ["-c:a", "libopus"])

    captured_kwargs: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Good afternoon, 17:00 works well for two."
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    service = VoiceService(
        api_key="sk-or-test",
        stt_model="google/gemini-3.6-flash",
        tts_voice="en-GB-RyanNeural",
        openai_client=fake_client,
    )

    transcript = await service.transcribe_voice(sample_ogg)
    assert transcript == "Good afternoon, 17:00 works well for two."
    assert captured_kwargs["model"] == "google/gemini-3.6-flash"

    content_parts = captured_kwargs["messages"][0]["content"]
    audio_part = next(p for p in content_parts if p["type"] == "input_audio")
    assert audio_part["input_audio"]["format"] == "wav"
    assert len(audio_part["input_audio"]["data"]) > 100


async def test_synthesize_voice_ogg_strips_markers_and_produces_opus_ogg(tmp_path: Path):
    seen_tts: dict = {}

    async def fake_tts_saver(text: str, voice: str, mp3_path: Path) -> None:
        seen_tts["text"] = text
        seen_tts["voice"] = voice
        await _generate_sample_audio(mp3_path, ["-c:a", "libmp3lame"])

    service = VoiceService(
        api_key="sk-or-test",
        tts_voice="en-GB-RyanNeural",
        tts_saver=fake_tts_saver,
    )

    out_ogg = tmp_path / "reply.ogg"
    result_path = await service.synthesize_voice_ogg(
        "Lovely, thank you ever so much!\u200b", out_ogg
    )

    assert result_path == out_ogg
    assert out_ogg.exists()
    assert out_ogg.stat().st_size > 200
    assert out_ogg.read_bytes()[:4] == b"OggS"
    assert seen_tts["text"] == "Lovely, thank you ever so much!"
    assert seen_tts["voice"] == "en-GB-RyanNeural"


async def test_synthesize_voice_ogg_rejects_empty_text(tmp_path: Path):
    service = VoiceService(api_key="sk-or-test")
    with pytest.raises(ValueError, match="empty"):
        await service.synthesize_voice_ogg("   \u200b  ", tmp_path / "empty.ogg")

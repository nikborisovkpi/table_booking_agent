import asyncio
import base64
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
import uuid

import edge_tts
from openai import AsyncOpenAI

from src.models import MessageModality


TTSSaver = Callable[[str, str, Path], Awaitable[None]]


def resolve_reply_modality(incoming_modality: MessageModality | None) -> MessageModality:
    """Mirror incoming message modality (defaulting to TEXT on conversation start)."""
    if incoming_modality == MessageModality.VOICE:
        return MessageModality.VOICE
    return MessageModality.TEXT


async def _default_edge_tts_saver(text: str, voice: str, mp3_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(mp3_path))


class VoiceService:
    """Handles Speech-to-Text (via OpenRouter multimodal model) and Text-to-Speech (Edge TTS + ffmpeg Opus)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        stt_model: str = "google/gemini-3.6-flash",
        tts_voice: str = "en-GB-RyanNeural",
        openai_client: Any | None = None,
        tts_saver: TTSSaver | None = None,
    ) -> None:
        self.stt_model = stt_model
        self.tts_voice = tts_voice
        self._client = openai_client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._tts_saver = tts_saver or _default_edge_tts_saver

    @staticmethod
    async def _run_ffmpeg(*args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {err_msg}")

    async def convert_audio_to_wav(self, input_path: Path, wav_path: Path) -> Path:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_ffmpeg(
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        )
        return wav_path

    async def convert_audio_to_telegram_ogg(self, input_path: Path, ogg_path: Path) -> Path:
        ogg_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_ffmpeg(
            "-i",
            str(input_path),
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-vbr",
            "on",
            "-compression_level",
            "10",
            "-frame_duration",
            "60",
            "-application",
            "voip",
            str(ogg_path),
        )
        return ogg_path

    async def transcribe_voice(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        temp_wav = audio_path.with_name(f"{audio_path.stem}_{uuid.uuid4().hex[:8]}.wav")
        try:
            await self.convert_audio_to_wav(audio_path, temp_wav)
            b64_wav = base64.b64encode(temp_wav.read_bytes()).decode("ascii")

            response = await self._client.chat.completions.create(
                model=self.stt_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe this voice message verbatim. "
                                    "Return ONLY the spoken words with no commentary or quotes."
                                ),
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": b64_wav,
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0,
            )

            raw_content = response.choices[0].message.content or ""
            transcript = raw_content.strip()
            if not transcript:
                raise RuntimeError("STT model returned an empty transcription")
            return transcript
        finally:
            temp_wav.unlink(missing_ok=True)

    async def synthesize_voice_ogg(self, text: str, output_ogg_path: Path) -> Path:
        cleaned_text = text.replace("\u200b", "").strip()
        if not cleaned_text:
            raise ValueError("Cannot synthesize voice from empty text")

        output_ogg_path = Path(output_ogg_path)
        output_ogg_path.parent.mkdir(parents=True, exist_ok=True)
        temp_mp3 = output_ogg_path.with_name(
            f"{output_ogg_path.stem}_{uuid.uuid4().hex[:8]}.mp3"
        )
        try:
            await self._tts_saver(cleaned_text, self.tts_voice, temp_mp3)
            return await self.convert_audio_to_telegram_ogg(temp_mp3, output_ogg_path)
        finally:
            temp_mp3.unlink(missing_ok=True)

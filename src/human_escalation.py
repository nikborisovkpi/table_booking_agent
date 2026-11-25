import asyncio
from pathlib import Path
from typing import Any
import uuid

from telethon import events

AGENT_MARKER_PREFIX = "🤖 [AGENT"
AGENT_INVISIBLE_TAG = "\u200b"


class TelegramOwnerBridge:
    """Escalates unknown questions from the booking agent to the owner's Saved Messages ('me')."""

    def __init__(
        self,
        *,
        client: Any,
        escalation_chat: str = "me",
        timeout_seconds: float = 300.0,
        notify_terminal: bool = True,
    ) -> None:
        self._client = client
        self.escalation_chat = escalation_chat
        self.timeout_seconds = timeout_seconds
        self.notify_terminal = notify_terminal

        self._lock = asyncio.Lock()
        self._pending_future: asyncio.Future[str] | None = None
        self._agent_message_ids: set[int] = set()

    def _is_agent_own_message(self, message_id: int, text: str | None) -> bool:
        if message_id in self._agent_message_ids:
            return True
        if not text:
            return False
        stripped = text.lstrip()
        return stripped.startswith(AGENT_MARKER_PREFIX) or text.endswith(AGENT_INVISIBLE_TAG)

    async def _send_agent_notice(self, body: str) -> None:
        payload = f"{body}{AGENT_INVISIBLE_TAG}"
        sent = await self._client.send_message(self.escalation_chat, payload)
        if sent is not None and getattr(sent, "id", None) is not None:
            self._agent_message_ids.add(sent.id)

    async def on_owner_chat_message(self, message_id: int, text: str | None) -> bool:
        if self._is_agent_own_message(message_id, text):
            return False
        if self._pending_future is None or self._pending_future.done():
            return False

        cleaned = (text or "").replace(AGENT_INVISIBLE_TAG, "").strip()
        if not cleaned:
            return False

        self._pending_future.set_result(cleaned)
        return True

    async def ask_owner(self, question: str) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question to owner must not be empty")

        async with self._lock:
            loop = asyncio.get_running_loop()
            self._pending_future = loop.create_future()
            try:
                prompt_msg = (
                    f"{AGENT_MARKER_PREFIX} QUESTION]\n"
                    f"Менеджер ресторану питає інформацію, якої я не знаю:\n\n"
                    f"❓ {clean_question}\n\n"
                    f"Напишіть відповідь просто сюди в цей чат."
                )
                if self.notify_terminal:
                    print(
                        f"\a\n[OWNER ESCALATION] Waiting for reply in Telegram ({self.escalation_chat}): "
                        f"{clean_question}",
                        flush=True,
                    )

                await self._send_agent_notice(prompt_msg)

                try:
                    answer = await asyncio.wait_for(
                        self._pending_future,
                        timeout=self.timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"Owner did not reply in '{self.escalation_chat}' within {self.timeout_seconds}s"
                    ) from exc

                await self._send_agent_notice(
                    f"{AGENT_MARKER_PREFIX} INFO] Прийнято: «{answer}». Продовжую діалог із менеджером."
                )
                return answer
            finally:
                self._pending_future = None

    def register_telethon_handler(
        self,
        *,
        voice_service: Any | None = None,
        temp_dir: Path = Path("tmp"),
    ) -> None:
        """Attaches a Telethon NewMessage listener to `self.escalation_chat`."""

        @self._client.on(events.NewMessage(chats=self.escalation_chat))
        async def _owner_listener(event: Any) -> None:
            msg = event.message
            raw_text = msg.raw_text or ""

            if self._is_agent_own_message(msg.id, raw_text):
                return
            if self._pending_future is None or self._pending_future.done():
                return

            if getattr(msg, "voice", None) and voice_service is not None:
                temp_dir.mkdir(parents=True, exist_ok=True)
                ogg_path = temp_dir / f"owner_voice_{uuid.uuid4().hex[:8]}.ogg"
                try:
                    await msg.download_media(file=str(ogg_path))
                    transcript = await voice_service.transcribe_voice(ogg_path)
                    await self.on_owner_chat_message(msg.id, transcript)
                finally:
                    ogg_path.unlink(missing_ok=True)
                return

            await self.on_owner_chat_message(msg.id, raw_text)

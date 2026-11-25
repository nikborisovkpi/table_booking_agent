import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

from telethon import TelegramClient, events

from src.config import Settings
from src.decision_logger import DecisionLogger
from src.human_escalation import AGENT_INVISIBLE_TAG, AGENT_MARKER_PREFIX
from src.models import AgentTurnOutput, MessageModality
from src.voice import resolve_reply_modality


BatchCallback = Callable[[str, MessageModality], Awaitable[None]]


@dataclass(frozen=True)
class IncomingItem:
    text: str
    modality: MessageModality


class MessageBatcher:
    """Coalesces consecutive Telegram messages into a single conversational turn."""

    def __init__(
        self,
        *,
        debounce_seconds: float,
        on_batch: BatchCallback,
        max_wait_seconds: float = 25.0,
    ) -> None:
        self.debounce_seconds = debounce_seconds
        self.max_wait_seconds = max_wait_seconds
        self._on_batch = on_batch

        self._items: list[IncomingItem] = []
        self._first_item_at: float | None = None
        self._deadline: float = 0.0
        self._timer_task: asyncio.Task[None] | None = None
        self._processing_lock = asyncio.Lock()
        self._closed = False

    async def push(self, item: IncomingItem) -> None:
        if self._closed:
            return
        cleaned = item.text.replace(AGENT_INVISIBLE_TAG, "").strip()
        if not cleaned:
            return

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._first_item_at is None:
            self._first_item_at = now

        self._items.append(IncomingItem(text=cleaned, modality=item.modality))
        self._bump_deadline(now)

        if self._timer_task is None or self._timer_task.done():
            self._timer_task = asyncio.create_task(self._wait_and_flush())

    def notify_typing(self) -> None:
        """Extends the current batch deadline while the manager is actively typing or recording."""
        if self._closed or not self._items or self._first_item_at is None:
            return
        now = asyncio.get_running_loop().time()
        self._bump_deadline(now)

    def _bump_deadline(self, now: float) -> None:
        assert self._first_item_at is not None
        hard_cap = self._first_item_at + self.max_wait_seconds
        self._deadline = min(now + self.debounce_seconds, hard_cap)

    async def _wait_and_flush(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed and self._items:
            while not self._closed:
                remaining = self._deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)

            if self._closed or not self._items:
                return

            batch = list(self._items)
            self._items.clear()
            self._first_item_at = None

            combined_text = "\n".join(i.text for i in batch)
            modality = (
                MessageModality.VOICE
                if any(i.modality == MessageModality.VOICE for i in batch)
                else MessageModality.TEXT
            )

            async with self._processing_lock:
                await self._on_batch(combined_text, modality)

    async def close(self) -> None:
        self._closed = True
        if self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass


async def resolve_target_entity(client: TelegramClient, target_chat: str) -> Any:
    """Resolves @username, numeric chat ID, 'me', or exact dialog title (for private test groups)."""
    cleaned = target_chat.strip()
    if not cleaned:
        raise ValueError("TARGET_CHAT must not be empty")

    # Try integer chat ID first if applicable
    try:
        numeric_id = int(cleaned)
    except ValueError:
        numeric_id = None

    if numeric_id is not None:
        try:
            return await client.get_entity(numeric_id)
        except Exception:
            async for dialog in client.iter_dialogs():
                if dialog.id == numeric_id:
                    return dialog.entity
            raise ValueError(f"Could not find Telegram dialog with numeric ID {numeric_id}")

    if cleaned.startswith(("@", "https://t.me/", "+")) or cleaned == "me":
        return await client.get_entity(cleaned)

    # Match by dialog display title (ideal for private 1-person test group)
    lowered = cleaned.lower()
    async for dialog in client.iter_dialogs():
        if (dialog.name or "").strip().lower() == lowered:
            return dialog.entity

    # Final fallback to Telethon's native resolver
    return await client.get_entity(cleaned)


class TelegramBookingRunner:
    """Coordinates Telethon I/O, message debouncing, voice mirroring, logging, and clean completion."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: Any,
        target_entity: Any,
        agent: Any,
        voice_service: Any,
        decision_logger: DecisionLogger,
        temp_dir: Path = Path("tmp"),
    ) -> None:
        self.settings = settings
        self._client = client
        self.target_entity = target_entity
        self._agent = agent
        self._voice_service = voice_service
        self._logger = decision_logger
        self._temp_dir = Path(temp_dir)

        self._outbound_send_lock = asyncio.Lock()
        self._agent_sent_ids: set[int] = set()
        self.completed_event = asyncio.Event()

        self._batcher = MessageBatcher(
            debounce_seconds=settings.debounce_seconds,
            on_batch=self._handle_batched_manager_turn,
        )

    async def _send_reply(self, text: str, modality: MessageModality) -> None:
        if modality == MessageModality.VOICE:
            self._temp_dir.mkdir(parents=True, exist_ok=True)
            ogg_path = self._temp_dir / f"agent_reply_{uuid.uuid4().hex[:8]}.ogg"
            try:
                async with self._client.action(self.target_entity, "record-audio"):
                    await self._voice_service.synthesize_voice_ogg(text, ogg_path)
                    async with self._outbound_send_lock:
                        sent = await self._client.send_file(
                            self.target_entity,
                            str(ogg_path),
                            voice_note=True,
                        )
                        if sent is not None and getattr(sent, "id", None) is not None:
                            self._agent_sent_ids.add(sent.id)
            finally:
                ogg_path.unlink(missing_ok=True)
            return

        async with self._client.action(self.target_entity, "typing"):
            await asyncio.sleep(0.4)
            async with self._outbound_send_lock:
                sent = await self._client.send_message(self.target_entity, text)
                if sent is not None and getattr(sent, "id", None) is not None:
                    self._agent_sent_ids.add(sent.id)

    async def initiate_opening_turn(self) -> AgentTurnOutput:
        turn, consultations = await self._agent.start_conversation()
        out_mod = resolve_reply_modality(None)
        await self._send_reply(turn.reply_text, out_mod)
        log_line = self._logger.log_turn(
            turn=turn,
            incoming_text=None,
            incoming_modality=None,
            outgoing_modality=out_mod,
            owner_consultations=consultations,
        )
        print(f"[TURN LOG] {log_line}", flush=True)
        return turn

    async def handle_raw_target_message(
        self,
        *,
        message_id: int,
        raw_text: str,
        is_voice: bool,
        download_fn: Callable[[str], Awaitable[Any]] | None,
    ) -> None:
        if self.completed_event.is_set():
            return

        # Wait if an outbound send is currently in flight so `_agent_sent_ids` is up to date
        async with self._outbound_send_lock:
            if message_id in self._agent_sent_ids:
                return

        if raw_text.lstrip().startswith(AGENT_MARKER_PREFIX) or raw_text.endswith(AGENT_INVISIBLE_TAG):
            return

        await self._client.send_read_acknowledge(self.target_entity)

        if is_voice and download_fn is not None:
            self._temp_dir.mkdir(parents=True, exist_ok=True)
            in_ogg = self._temp_dir / f"manager_in_{uuid.uuid4().hex[:8]}.ogg"
            try:
                await download_fn(str(in_ogg))
                transcript = await self._voice_service.transcribe_voice(in_ogg)
            finally:
                in_ogg.unlink(missing_ok=True)
            print(f"[INCOMING VOICE TRANSCRIPT] {transcript}", flush=True)
            await self._batcher.push(
                IncomingItem(text=transcript, modality=MessageModality.VOICE)
            )
            return

        if raw_text.strip():
            print(f"[INCOMING TEXT] {raw_text.strip()}", flush=True)
            await self._batcher.push(
                IncomingItem(text=raw_text, modality=MessageModality.TEXT)
            )

    async def _handle_batched_manager_turn(
        self, combined_text: str, incoming_modality: MessageModality
    ) -> None:
        if self.completed_event.is_set():
            return

        turn, consultations = await self._agent.process_manager_message(
            combined_text,
            modality=incoming_modality,
        )
        outgoing_modality = resolve_reply_modality(incoming_modality)
        await self._send_reply(turn.reply_text, outgoing_modality)

        log_line = self._logger.log_turn(
            turn=turn,
            incoming_text=combined_text,
            incoming_modality=incoming_modality,
            outgoing_modality=outgoing_modality,
            owner_consultations=consultations,
        )
        print(f"[TURN LOG] {log_line}", flush=True)

        if turn.booking_confirmed:
            details = turn.confirmed_details
            print(
                "\n========================================\n"
                "✅ BOOKING CONFIRMED!\n"
                f"Restaurant : {details.restaurant_name if details else 'N/A'}\n"
                f"Date       : {details.date if details else 'N/A'}\n"
                f"Time       : {details.time if details else 'N/A'}\n"
                f"Party size : {details.party_size if details else 'N/A'}\n"
                f"Decision log: {self.settings.decision_log_path}\n"
                "========================================\n",
                flush=True,
            )
            self.completed_event.set()

    def register_telethon_handlers(self) -> None:
        incoming_filter = None if self.settings.self_test_mode else True

        @self._client.on(
            events.NewMessage(chats=self.target_entity, incoming=incoming_filter)
        )
        async def _on_target_msg(event: Any) -> None:
            msg = event.message
            is_voice = bool(getattr(msg, "voice", None))
            await self.handle_raw_target_message(
                message_id=msg.id,
                raw_text=msg.raw_text or "",
                is_voice=is_voice,
                download_fn=(lambda path: msg.download_media(file=path)) if is_voice else None,
            )

        @self._client.on(events.UserUpdate(chats=self.target_entity))
        async def _on_user_typing(event: Any) -> None:
            if getattr(event, "typing", False) or getattr(event, "recording", False):
                self._batcher.notify_typing()

    async def close(self) -> None:
        await self._batcher.close()

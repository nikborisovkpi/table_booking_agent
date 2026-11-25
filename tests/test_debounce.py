import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.config import Settings
from src.decision_logger import DecisionLogger
from src.models import AgentTurnOutput, ConfirmedBookingDetails, MessageModality
from src.telegram_client import IncomingItem, MessageBatcher, TelegramBookingRunner


async def test_message_batcher_coalesces_burst_and_mirrors_voice_modality():
    flushed_batches: list[tuple[str, MessageModality]] = []

    async def on_flush(combined_text: str, modality: MessageModality) -> None:
        flushed_batches.append((combined_text, modality))

    batcher = MessageBatcher(debounce_seconds=0.06, on_batch=on_flush)

    await batcher.push(IncomingItem(text="Good afternoon!", modality=MessageModality.TEXT))
    await asyncio.sleep(0.02)
    await batcher.push(
        IncomingItem(text="Yes, 17:00 is free at Dishoom.", modality=MessageModality.VOICE)
    )

    await asyncio.sleep(0.12)
    await batcher.close()

    assert len(flushed_batches) == 1
    combined_text, modality = flushed_batches[0]
    assert combined_text == "Good afternoon!\nYes, 17:00 is free at Dishoom."
    assert modality == MessageModality.VOICE


async def test_message_batcher_extends_wait_on_typing_activity():
    flushed_at: list[float] = []
    loop = asyncio.get_running_loop()

    async def on_flush(combined_text: str, modality: MessageModality) -> None:
        flushed_at.append(loop.time())

    batcher = MessageBatcher(debounce_seconds=0.08, max_wait_seconds=0.5, on_batch=on_flush)
    t0 = loop.time()

    await batcher.push(IncomingItem(text="Hold on...", modality=MessageModality.TEXT))
    await asyncio.sleep(0.05)
    batcher.notify_typing()
    await asyncio.sleep(0.05)
    batcher.notify_typing()

    await asyncio.sleep(0.15)
    await batcher.close()

    assert len(flushed_at) == 1
    assert (flushed_at[0] - t0) >= 0.15


async def test_message_batcher_flushes_messages_arriving_while_previous_batch_is_processing():
    batches: list[str] = []
    first_batch_entered = asyncio.Event()
    release_first_batch = asyncio.Event()

    async def on_flush(combined_text: str, modality: MessageModality) -> None:
        batches.append(combined_text)
        if len(batches) == 1:
            first_batch_entered.set()
            await release_first_batch.wait()

    batcher = MessageBatcher(debounce_seconds=0.04, on_batch=on_flush)

    await batcher.push(IncomingItem(text="Batch 1", modality=MessageModality.TEXT))
    await asyncio.wait_for(first_batch_entered.wait(), timeout=1.0)

    # Arrives while Batch 1 is still blocked inside on_flush (e.g. waiting for owner reply)
    await batcher.push(IncomingItem(text="Batch 2 follow-up", modality=MessageModality.TEXT))
    release_first_batch.set()

    await asyncio.sleep(0.12)
    await batcher.close()

    assert batches == ["Batch 1", "Batch 2 follow-up"]


async def test_runner_ignores_own_inflight_messages_in_self_test_mode_and_completes_on_confirmation(
    tmp_path: Path,
):
    log_path = tmp_path / "decision.log"
    settings = Settings(
        _env_file=None,
        telegram_api_id=31939235,
        telegram_api_hash="test_hash",
        openrouter_api_key="sk-or-test",
        self_test_mode=True,
        debounce_seconds=0.04,
        decision_log_path=log_path,
    )

    runner_ref: dict[str, TelegramBookingRunner] = {}
    sent_records: list[tuple[str, str]] = []
    next_msg_id = 0

    class FakeTelethonClient:
        async def send_read_acknowledge(self, entity):
            return None

        def action(self, entity, action_name: str):
            class _NoopCtx:
                async def __aenter__(self_inner):
                    return None

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return _NoopCtx()

        async def send_message(self, entity, text: str):
            nonlocal next_msg_id
            next_msg_id += 1
            mid = next_msg_id
            sent_records.append(("text", text))
            # Fire mid-flight incoming update BEFORE send_message returns!
            asyncio.create_task(
                runner_ref["runner"].handle_raw_target_message(
                    message_id=mid,
                    raw_text=text,
                    is_voice=False,
                    download_fn=None,
                )
            )
            await asyncio.sleep(0.01)
            return SimpleNamespace(id=mid)

        async def send_file(self, entity, file: str, *, voice_note: bool = False):
            nonlocal next_msg_id
            next_msg_id += 1
            mid = next_msg_id
            sent_records.append(("voice", str(file)))
            asyncio.create_task(
                runner_ref["runner"].handle_raw_target_message(
                    message_id=mid,
                    raw_text="",
                    is_voice=True,
                    download_fn=None,
                )
            )
            await asyncio.sleep(0.01)
            return SimpleNamespace(id=mid)

    class FakeAgent:
        async def start_conversation(self):
            return (
                AgentTurnOutput(
                    understood="Starting booking chat.",
                    decision="Greet politely and request table for 2 at 17:00 today.",
                    reply_text="Good afternoon! Could we book a table for two at 17:00 today under Nikita, please?",
                ),
                [],
            )

        async def process_manager_message(self, text: str, *, modality: MessageModality):
            return (
                AgentTurnOutput(
                    understood="Manager confirmed 17:00 today at Sketch London.",
                    decision="Send final confirmation repeating all four details.",
                    reply_text=(
                        "Lovely, thank you! Confirming a table for 2 at 17:00 today, "
                        "Wednesday 25 November 2026, at Sketch London under Nikita."
                    ),
                    booking_confirmed=True,
                    confirmed_details=ConfirmedBookingDetails(
                        time="17:00",
                        date="Wednesday, 25 November 2026",
                        party_size=2,
                        restaurant_name="Sketch London",
                    ),
                ),
                [],
            )

    class FakeVoiceService:
        async def transcribe_voice(self, path: Path) -> str:
            return "All booked at Sketch London for 2 at 17:00 today!"

        async def synthesize_voice_ogg(self, text: str, out_path: Path) -> Path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"OggSfake")
            return out_path

    runner = TelegramBookingRunner(
        settings=settings,
        client=FakeTelethonClient(),
        target_entity="test_group",
        agent=FakeAgent(),
        voice_service=FakeVoiceService(),
        decision_logger=DecisionLogger(log_path),
        temp_dir=tmp_path / "tmp_audio",
    )
    runner_ref["runner"] = runner

    await runner.initiate_opening_turn()
    # Allow mid-flight echo task to run and verify it did NOT trigger a second turn
    await asyncio.sleep(0.08)
    assert len(sent_records) == 1
    assert sent_records[0][0] == "text"

    # Now simulate manager sending a VOICE message in the same self-test group
    async def fake_download(file: str):
        Path(file).write_bytes(b"OggSincoming")

    await runner.handle_raw_target_message(
        message_id=777,
        raw_text="",
        is_voice=True,
        download_fn=fake_download,
    )

    await asyncio.wait_for(runner.completed_event.wait(), timeout=1.0)
    await runner.close()

    assert len(sent_records) == 2
    assert sent_records[1][0] == "voice"  # Mirrored voice -> voice!

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 2
    assert "SENT[TEXT]:" in log_lines[0]
    assert "IN[VOICE]:" in log_lines[1]
    assert "SENT[VOICE]:" in log_lines[1]
    assert "STATUS: CONFIRMED" in log_lines[1]

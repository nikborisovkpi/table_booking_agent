import asyncio
from types import SimpleNamespace
import pytest

from src.human_escalation import AGENT_MARKER_PREFIX, TelegramOwnerBridge


async def test_owner_bridge_ignores_own_messages_even_during_send_race_and_returns_owner_reply():
    sent_messages: list[tuple[str, str]] = []
    bridge_holder: dict[str, TelegramOwnerBridge] = {}
    next_id = 100

    class FakeTelegramClient:
        async def send_message(self, entity: str, text: str):
            nonlocal next_id
            next_id += 1
            msg_id = next_id
            sent_messages.append((entity, text))
            # Simulate Telethon firing NewMessage BEFORE send_message returns msg.id!
            await bridge_holder["bridge"].on_owner_chat_message(
                message_id=msg_id,
                text=text,
            )
            return SimpleNamespace(id=msg_id)

    bridge = TelegramOwnerBridge(
        client=FakeTelegramClient(),
        escalation_chat="me",
        timeout_seconds=2.0,
        notify_terminal=False,
    )
    bridge_holder["bridge"] = bridge

    async def simulate_human_typing_later():
        await asyncio.sleep(0.05)
        await bridge.on_owner_chat_message(
            message_id=999,
            text="  Yes, 17:30 works fine, no allergies.  ",
        )

    human_task = asyncio.create_task(simulate_human_typing_later())
    answer = await bridge.ask_owner("Manager says 17:00 is full; is 17:30 acceptable?")
    await human_task

    assert answer == "Yes, 17:30 works fine, no allergies."
    assert sent_messages[0][0] == "me"
    assert sent_messages[0][1].startswith(AGENT_MARKER_PREFIX)
    assert "17:30 acceptable?" in sent_messages[0][1]


async def test_owner_bridge_ignores_stray_saved_messages_when_idle():
    class FakeTelegramClient:
        async def send_message(self, entity: str, text: str):
            return SimpleNamespace(id=1)

    bridge = TelegramOwnerBridge(
        client=FakeTelegramClient(),
        escalation_chat="me",
        notify_terminal=False,
    )
    accepted = await bridge.on_owner_chat_message(
        message_id=42,
        text="Shopping list: milk, bread",
    )
    assert accepted is False


async def test_owner_bridge_serializes_concurrent_questions():
    order: list[str] = []
    bridge_holder: dict[str, TelegramOwnerBridge] = {}
    msg_counter = 0

    class FakeTelegramClient:
        async def send_message(self, entity: str, text: str):
            nonlocal msg_counter
            msg_counter += 1
            if "Question 1" in text:
                order.append("sent_q1")
                asyncio.get_running_loop().call_later(
                    0.03,
                    lambda: asyncio.create_task(
                        bridge_holder["bridge"].on_owner_chat_message(501, "Answer 1")
                    ),
                )
            elif "Question 2" in text:
                order.append("sent_q2")
                asyncio.get_running_loop().call_later(
                    0.03,
                    lambda: asyncio.create_task(
                        bridge_holder["bridge"].on_owner_chat_message(502, "Answer 2")
                    ),
                )
            return SimpleNamespace(id=msg_counter)

    bridge = TelegramOwnerBridge(
        client=FakeTelegramClient(),
        escalation_chat="me",
        timeout_seconds=2.0,
        notify_terminal=False,
    )
    bridge_holder["bridge"] = bridge

    ans1, ans2 = await asyncio.gather(
        bridge.ask_owner("Question 1: phone number?"),
        bridge.ask_owner("Question 2: indoor or terrace?"),
    )

    assert ans1 == "Answer 1"
    assert ans2 == "Answer 2"
    assert order == ["sent_q1", "sent_q2"]


async def test_owner_bridge_raises_timeout_when_owner_does_not_reply():
    class FakeTelegramClient:
        async def send_message(self, entity: str, text: str):
            return SimpleNamespace(id=1)

    bridge = TelegramOwnerBridge(
        client=FakeTelegramClient(),
        escalation_chat="me",
        timeout_seconds=0.05,
        notify_terminal=False,
    )
    with pytest.raises(TimeoutError, match="Owner did not reply"):
        await bridge.ask_owner("Do you want a window table?")

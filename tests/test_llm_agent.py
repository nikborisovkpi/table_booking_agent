from datetime import datetime, timezone
import json
from types import SimpleNamespace

from src.config import Settings
from src.llm_agent import BookingAgent
from src.models import MessageModality
from src.prompts import build_system_prompt


def _make_settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_api_id=31939235,
        telegram_api_hash="test_hash",
        openrouter_api_key="sk-or-test",
        openrouter_model="anthropic/claude-haiku-5.5",
        guest_name="Nikita",
        party_size=2,
        booking_time="17:00",
    )


def _tool_call_response(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            type="function",
                            function=SimpleNamespace(
                                name=name,
                                arguments=json.dumps(arguments),
                            ),
                        )
                    ],
                )
            )
        ]
    )


def test_build_system_prompt_contains_known_facts_and_anti_hallucination_rules():
    settings = _make_settings()
    sample_now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    prompt = build_system_prompt(settings, now=sample_now)

    assert "Nikita" in prompt
    assert "17:00" in prompt
    assert "Wednesday, 25 November 2026" in prompt
    assert "British English" in prompt
    assert "ask_owner" in prompt
    assert "NEVER invent" in prompt or "never invent" in prompt.lower()
    assert "restaurant" in prompt.lower()


async def test_booking_agent_initiates_conversation():
    settings = _make_settings()
    calls: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return _tool_call_response(
                "call_init",
                "submit_turn",
                {
                    "understood": "Starting conversation with the restaurant manager.",
                    "decision": "Send polite British opening asking for a table for 2 at 17:00 today.",
                    "reply_text": (
                        "Good afternoon! I was wondering if we might book a table for two "
                        "at 17:00 today under the name Nikita, please?"
                    ),
                    "booking_confirmed": False,
                    "confirmed_details": None,
                },
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    async def unused_ask_owner(q: str) -> str:
        raise AssertionError(f"Should not ask owner on greeting: {q}")

    agent = BookingAgent(
        settings=settings,
        ask_owner_fn=unused_ask_owner,
        openai_client=fake_client,
    )
    turn, consultations = await agent.start_conversation()

    assert consultations == []
    assert turn.booking_confirmed is False
    assert "Good afternoon" in turn.reply_text
    assert calls[0]["model"] == "anthropic/claude-haiku-5.5"


async def test_booking_agent_calls_ask_owner_for_unknown_facts_and_records_consultation():
    settings = _make_settings()
    step = 0

    class FakeCompletions:
        async def create(self, **kwargs):
            nonlocal step
            step += 1
            if step == 1:
                return _tool_call_response(
                    "call_ask_1",
                    "ask_owner",
                    {
                        "question": "На 17:00 місць немає, менеджер пропонує 17:30 і питає про алергії. Підходить?"
                    },
                )
            # Verify the owner's answer was injected into messages
            tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
            assert any("17:30 підходить, алергій немає" in m["content"] for m in tool_msgs)
            return _tool_call_response(
                "call_submit_1",
                "submit_turn",
                {
                    "understood": "Manager offered 17:30 instead of 17:00 and asked about allergies.",
                    "decision": "Consulted owner; accept 17:30, state no allergies, and ask restaurant name.",
                    "reply_text": (
                        "Half past five works wonderfully for us, thank you, and neither of us has any allergies. "
                        "Could I just double-check the name of the restaurant as well, please?"
                    ),
                    "booking_confirmed": False,
                    "confirmed_details": None,
                },
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    asked_questions: list[str] = []

    async def fake_ask_owner(question: str) -> str:
        asked_questions.append(question)
        return "17:30 підходить, алергій немає"

    agent = BookingAgent(
        settings=settings,
        ask_owner_fn=fake_ask_owner,
        openai_client=fake_client,
    )
    turn, consultations = await agent.process_manager_message(
        "Sorry, 17:00 is fully booked! Would 17:30 work? Any allergies?",
        modality=MessageModality.VOICE,
    )

    assert len(asked_questions) == 1
    assert consultations == [
        (
            "На 17:00 місць немає, менеджер пропонує 17:30 і питає про алергії. Підходить?",
            "17:30 підходить, алергій немає",
        )
    ]
    assert "Half past five" in turn.reply_text


async def test_booking_agent_rejects_premature_confirmation_missing_restaurant_in_reply_and_self_corrects():
    settings = _make_settings()
    step = 0

    class FakeCompletions:
        async def create(self, **kwargs):
            nonlocal step
            step += 1
            if step == 1:
                # Premature confirmation: reply_text omits restaurant name!
                return _tool_call_response(
                    "call_bad_confirm",
                    "submit_turn",
                    {
                        "understood": "Manager confirmed booking at Dishoom Covent Garden.",
                        "decision": "Confirm booking.",
                        "reply_text": "Lovely, thank you! See you at 17:00 today for two.",
                        "booking_confirmed": True,
                        "confirmed_details": {
                            "time": "17:00",
                            "date": "Wednesday, 25 November 2026",
                            "party_size": 2,
                            "restaurant_name": "Dishoom Covent Garden",
                        },
                    },
                )
            # Step 2: verify validation feedback reached the LLM, then submit proper confirmation
            tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
            assert any("Dishoom Covent Garden" in m["content"] for m in tool_msgs)
            return _tool_call_response(
                "call_good_confirm",
                "submit_turn",
                {
                    "understood": "Manager confirmed booking at Dishoom Covent Garden.",
                    "decision": "Repeat all 4 confirmed details clearly and conclude.",
                    "reply_text": (
                        "Wonderful, thank you ever so much! Just to confirm: a table for 2 "
                        "at 17:00 today, Wednesday 25 November 2026, at Dishoom Covent Garden under Nikita."
                    ),
                    "booking_confirmed": True,
                    "confirmed_details": {
                        "time": "17:00",
                        "date": "Wednesday, 25 November 2026",
                        "party_size": 2,
                        "restaurant_name": "Dishoom Covent Garden",
                    },
                },
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    async def unused_ask_owner(q: str) -> str:
        return "unused"

    agent = BookingAgent(
        settings=settings,
        ask_owner_fn=unused_ask_owner,
        openai_client=fake_client,
    )
    turn, _ = await agent.process_manager_message(
        "All booked for you at Dishoom Covent Garden for 2 at 17:00 today!"
    )

    assert step == 2
    assert turn.booking_confirmed is True
    assert "Dishoom Covent Garden" in turn.reply_text

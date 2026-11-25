from collections.abc import Awaitable, Callable
from datetime import datetime
import json
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from src.config import Settings
from src.models import AgentTurnOutput, MessageModality
from src.prompts import build_system_prompt

AskOwnerFn = Callable[[str], Awaitable[str]]


ASK_OWNER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_owner",
        "description": (
            "Ask the human guest owner (Nikita) in Ukrainian for any fact, preference, "
            "contact detail, allergy info, or approval of an alternative time/condition "
            "that is NOT in KNOWN FACTS. Never guess; call this tool instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Clear, concise question in Ukrainian for the owner.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_TURN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_turn",
        "description": "Submit your structured decision and the British English reply to send to the manager.",
        "parameters": {
            "type": "object",
            "properties": {
                "understood": {
                    "type": "string",
                    "description": "One-sentence summary of what the manager communicated (or initial action).",
                },
                "decision": {
                    "type": "string",
                    "description": "One-sentence summary of what you decided to do next and why.",
                },
                "reply_text": {
                    "type": "string",
                    "description": "Polite British English message to send to the manager in Telegram.",
                },
                "booking_confirmed": {
                    "type": "boolean",
                    "description": (
                        "Set true ONLY when manager agreed AND you know the exact restaurant name "
                        "AND reply_text explicitly repeats time, date, party size, and restaurant name."
                    ),
                },
                "confirmed_details": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string"},
                                "date": {"type": "string"},
                                "party_size": {"type": "integer"},
                                "restaurant_name": {"type": "string"},
                            },
                            "required": ["time", "date", "party_size", "restaurant_name"],
                            "additionalProperties": False,
                        },
                    ]
                },
            },
            "required": ["understood", "decision", "reply_text", "booking_confirmed"],
            "additionalProperties": False,
        },
    },
}


class BookingAgent:
    """Manages conversation history, OpenRouter LLM calls, owner escalation tool loop, and confirmation validation."""

    def __init__(
        self,
        *,
        settings: Settings,
        ask_owner_fn: AskOwnerFn,
        openai_client: Any | None = None,
        manager_context: str | None = None,
        now_provider: Callable[[], datetime] | None = None,
        max_tool_steps: int = 8,
    ) -> None:
        self.settings = settings
        self._ask_owner_fn = ask_owner_fn
        self._client = openai_client or AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        self._now_provider = now_provider
        self._manager_context = manager_context
        self._max_tool_steps = max_tool_steps
        self._history: list[dict[str, Any]] = []

    def _current_system_prompt(self) -> str:
        now = self._now_provider() if self._now_provider else None
        return build_system_prompt(
            self.settings,
            now=now,
            manager_context=self._manager_context,
        )

    @staticmethod
    def _validate_confirmed_reply_content(turn: AgentTurnOutput) -> str | None:
        """Ensures Requirement 4: when booking_confirmed=True, reply_text repeats restaurant, time, date, and party size."""
        if not turn.booking_confirmed or turn.confirmed_details is None:
            return None

        reply_lower = turn.reply_text.lower()
        details = turn.confirmed_details

        missing: list[str] = []
        if details.restaurant_name.lower() not in reply_lower:
            missing.append(f"restaurant name ('{details.restaurant_name}')")

        # Check party size (either digit '2' or word 'two' etc.)
        word_numbers = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        size_tokens = {str(details.party_size)}
        if details.party_size in word_numbers:
            size_tokens.add(word_numbers[details.party_size])
        if not any(tok in reply_lower for tok in size_tokens):
            missing.append(f"number of people ({details.party_size})")

        if missing:
            return (
                "Requirement 4 check failed: when booking_confirmed=true, your `reply_text` MUST "
                f"explicitly repeat {', '.join(missing)}, along with the confirmed time and date."
            )
        return None

    @staticmethod
    def _extract_extra_content(obj: Any) -> Any:
        extra_content = getattr(obj, "extra_content", None)
        if extra_content is None:
            model_extra = getattr(obj, "model_extra", None)
            if isinstance(model_extra, dict):
                extra_content = model_extra.get("extra_content")
        return extra_content

    @classmethod
    def _serialize_tool_call(cls, tc: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        extra_content = cls._extract_extra_content(tc)
        if extra_content is not None:
            entry["extra_content"] = extra_content
        return entry

    async def _run_turn_loop(
        self, working_messages: list[dict[str, Any]]
    ) -> tuple[AgentTurnOutput, list[tuple[str, str]]]:
        consultations: list[tuple[str, str]] = []

        for _ in range(self._max_tool_steps):
            response = await self._client.chat.completions.create(
                model=self.settings.openrouter_model,
                messages=[
                    {"role": "system", "content": self._current_system_prompt()},
                    *working_messages,
                ],
                tools=[ASK_OWNER_TOOL, SUBMIT_TURN_TOOL],
                tool_choice="required",
                temperature=0.2,
            )

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                raw_content = (getattr(msg, "content", None) or "").strip()
                if raw_content:
                    try:
                        parsed = AgentTurnOutput.model_validate_json(raw_content)
                        err = self._validate_confirmed_reply_content(parsed)
                        if err is None:
                            return parsed, consultations
                    except ValidationError:
                        pass
                working_messages.append({"role": "assistant", "content": raw_content or "..."})
                working_messages.append(
                    {
                        "role": "user",
                        "content": "Please call the `submit_turn` tool to provide your structured response.",
                    }
                )
                continue

            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": getattr(msg, "content", None),
                "tool_calls": [self._serialize_tool_call(tc) for tc in tool_calls],
            }
            msg_extra = self._extract_extra_content(msg)
            if msg_extra is not None:
                assistant_entry["extra_content"] = msg_extra
            working_messages.append(assistant_entry)

            accepted_turn: AgentTurnOutput | None = None

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Invalid JSON arguments: {exc}",
                        }
                    )
                    continue

                if fn_name == "ask_owner":
                    question = str(args.get("question", "")).strip()
                    owner_answer = await self._ask_owner_fn(question)
                    consultations.append((question, owner_answer))
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Owner ({self.settings.guest_name}) answered: {owner_answer}",
                        }
                    )
                elif fn_name == "submit_turn":
                    try:
                        candidate = AgentTurnOutput.model_validate(args)
                    except ValidationError as exc:
                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": (
                                    f"Validation error in submit_turn: {exc}. "
                                    "If you do not yet know the exact restaurant name, set booking_confirmed=false "
                                    "and politely ask the manager for the restaurant's name first."
                                ),
                            }
                        )
                        continue

                    content_err = self._validate_confirmed_reply_content(candidate)
                    if content_err is not None:
                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content_err,
                            }
                        )
                        continue

                    accepted_turn = candidate
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Turn accepted.",
                        }
                    )
                else:
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Unknown tool '{fn_name}'. Use ask_owner or submit_turn.",
                        }
                    )

            if accepted_turn is not None:
                return accepted_turn, consultations

        raise RuntimeError(
            f"BookingAgent exceeded max_tool_steps ({self._max_tool_steps}) without a valid submit_turn"
        )

    async def start_conversation(self) -> tuple[AgentTurnOutput, list[tuple[str, str]]]:
        seed_prompt = {
            "role": "user",
            "content": (
                "[SYSTEM EVENT]: The conversation has not started yet, and the manager will not message first. "
                "Write the opening message in polite British English to request the table."
            ),
        }
        working = [*self._history, seed_prompt]
        turn, consultations = await self._run_turn_loop(working)
        self._history.append({"role": "assistant", "content": turn.reply_text})
        return turn, consultations

    async def process_manager_message(
        self,
        text: str,
        *,
        modality: MessageModality = MessageModality.TEXT,
    ) -> tuple[AgentTurnOutput, list[tuple[str, str]]]:
        clean_text = text.replace("\u200b", "").strip()
        user_entry = {
            "role": "user",
            "content": f"[Manager via {modality.value}]: {clean_text}",
        }
        self._history.append(user_entry)
        working = list(self._history)
        turn, consultations = await self._run_turn_loop(working)
        self._history.append({"role": "assistant", "content": turn.reply_text})
        return turn, consultations

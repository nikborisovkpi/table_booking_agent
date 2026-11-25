from datetime import datetime, timezone
from pathlib import Path

from src.decision_logger import DecisionLogger
from src.models import AgentTurnOutput, MessageModality


def test_decision_logger_writes_strictly_one_line_per_turn_even_with_multiline_messages(tmp_path: Path):
    log_file = tmp_path / "nested" / "decision.log"
    logger = DecisionLogger(log_file)
    fixed_ts = datetime(2026, 11, 25, 15, 5, 30, tzinfo=timezone.utc)

    # Turn 1: Agent initiates conversation (no incoming message)
    init_turn = AgentTurnOutput(
        understood="Starting a new conversation with the London restaurant manager.",
        decision="Politely request a table for 2 at 17:00 today under Nikita.",
        reply_text="Good afternoon!\nWould you happen to have a table for two at 17:00 today?\u200b",
    )
    line1 = logger.log_turn(
        turn=init_turn,
        incoming_text=None,
        incoming_modality=None,
        outgoing_modality=MessageModality.TEXT,
        timestamp=fixed_ts,
    )

    # Turn 2: Manager replies with multiline voice transcript + owner consultation
    reply_turn = AgentTurnOutput(
        understood="Manager says 17:00 is booked\nand offers 17:30.",
        decision="Owner approved 17:30; accept 17:30 and ask for the restaurant's name.",
        reply_text="Half past five suits us wonderfully!\r\nCould I also check the exact name of the restaurant?",
    )
    line2 = logger.log_turn(
        turn=reply_turn,
        incoming_text="Hello!\n17:00 is full | would 17:30 work?",
        incoming_modality=MessageModality.VOICE,
        outgoing_modality=MessageModality.VOICE,
        owner_consultations=[("Is 17:30 acceptable?", "Yes, 17:30 is fine")],
        timestamp=fixed_ts,
    )

    assert "\n" not in line1
    assert "\u200b" not in line1
    assert "\n" not in line2

    raw_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    assert "IN[NONE]" in raw_lines[0]
    assert "UNDERSTOOD:" in raw_lines[0]
    assert "DECIDED:" in raw_lines[0]
    assert "SENT[TEXT]:" in raw_lines[0]

    assert "IN[VOICE]:" in raw_lines[1]
    assert "SENT[VOICE]:" in raw_lines[1]
    assert "Is 17:30 acceptable?" in raw_lines[1]
    assert "Yes, 17:30 is fine" in raw_lines[1]

from datetime import datetime, timezone
from pathlib import Path
import re

from src.models import AgentTurnOutput, MessageModality


class DecisionLogger:
    """Writes strictly one single-line log record per conversation turn."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean(text: str) -> str:
        without_markers = text.replace("\u200b", "").replace("|", "/")
        collapsed = re.sub(r"[\r\n\t]+", " ", without_markers)
        collapsed = re.sub(r" {2,}", " ", collapsed)
        return collapsed.strip()

    def log_turn(
        self,
        *,
        turn: AgentTurnOutput,
        incoming_text: str | None = None,
        incoming_modality: MessageModality | None = None,
        outgoing_modality: MessageModality = MessageModality.TEXT,
        owner_consultations: list[tuple[str, str]] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat(timespec="seconds")

        if incoming_text is None:
            in_part = "IN[NONE]: <conversation initiated by agent>"
        else:
            mod = incoming_modality.value if incoming_modality else MessageModality.TEXT.value
            in_part = f'IN[{mod}]: "{self._clean(incoming_text)}"'

        understood_part = f"UNDERSTOOD: {self._clean(turn.understood)}"

        decision_text = self._clean(turn.decision)
        if owner_consultations:
            qa_str = "; ".join(
                f'Q: "{self._clean(q)}" -> A: "{self._clean(a)}"'
                for q, a in owner_consultations
            )
            decision_text = f"{decision_text} [Owner consulted: {qa_str}]"
        decided_part = f"DECIDED: {decision_text}"

        sent_part = f'SENT[{outgoing_modality.value}]: "{self._clean(turn.reply_text)}"'

        status_suffix = " | STATUS: CONFIRMED" if turn.booking_confirmed else ""
        line = f"[{ts}] | {in_part} | {understood_part} | {decided_part} | {sent_part}{status_suffix}"

        with self.log_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

        return line

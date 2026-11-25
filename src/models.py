from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


_PLACEHOLDER_RESTAURANT_VALUES = {
    "",
    "unknown",
    "n/a",
    "none",
    "tbd",
    "unspecified",
    "london restaurant",
    "a restaurant in london",
}


class MessageModality(str, Enum):
    TEXT = "TEXT"
    VOICE = "VOICE"


class ConfirmedBookingDetails(BaseModel):
    time: str = Field(..., min_length=1, description="Confirmed booking time, e.g. 17:00")
    date: str = Field(..., min_length=1, description="Confirmed booking date")
    party_size: int = Field(..., ge=1, description="Confirmed number of guests")
    restaurant_name: str = Field(..., min_length=1, description="Confirmed restaurant name")

    @field_validator("time", "date")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be blank")
        return cleaned

    @field_validator("restaurant_name")
    @classmethod
    def _validate_restaurant_name(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned.lower() in _PLACEHOLDER_RESTAURANT_VALUES:
            raise ValueError(
                "restaurant_name must be the actual confirmed restaurant name, not a placeholder"
            )
        return cleaned


class AgentTurnOutput(BaseModel):
    understood: str = Field(
        ...,
        min_length=1,
        description="Concise summary of what the manager communicated (or initial action)",
    )
    decision: str = Field(
        ...,
        min_length=1,
        description="What the agent decided to do next and why",
    )
    reply_text: str = Field(
        ...,
        min_length=1,
        description="Polite British English message to send to the manager",
    )
    booking_confirmed: bool = Field(
        default=False,
        description="True ONLY when the manager confirmed AND reply_text repeats time, date, party_size, and restaurant_name",
    )
    confirmed_details: ConfirmedBookingDetails | None = Field(
        default=None,
        description="Populated iff booking_confirmed is True",
    )

    @field_validator("understood", "decision", "reply_text")
    @classmethod
    def _strip_and_require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _require_details_when_confirmed(self) -> "AgentTurnOutput":
        if self.booking_confirmed and self.confirmed_details is None:
            raise ValueError(
                "confirmed_details (time, date, party_size, restaurant_name) is required when booking_confirmed is True"
            )
        return self

"""API contract (spec section 9) plus the route invariants (spec section 8.1) as executable code."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGE_CHARS = 2_000
MAX_SESSION_ID_CHARS = 128

Route = Literal["rag", "tool", "hybrid", "clarify", "no_answer", "refuse"]
ToolStatus = Literal["pending_confirmation", "success", "error"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, strict=True)

    session_id: str = Field(min_length=1, max_length=MAX_SESSION_ID_CHARS)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("session_id")
    @classmethod
    def _no_control_chars(cls, v: str) -> str:
        if any(ord(c) < 32 or ord(c) == 127 for c in v):
            raise ValueError("session_id contains control characters")
        return v


class Citation(BaseModel):
    doc_id: str
    locator: str


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: ToolStatus


class ChatResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer: str
    route: Route
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    pending_confirmation: bool = False


class RouteInvariantError(ValueError):
    pass


# route -> (must have citations, must have tool calls)
_ROUTE_RULES: dict[str, tuple[bool, bool]] = {
    "rag": (True, False),
    "tool": (False, True),
    "hybrid": (True, True),
    "clarify": (False, False),
    "no_answer": (False, False),
    "refuse": (False, False),
}


def validate_route_invariants(r: ChatResponse) -> None:
    want_cites, want_tools = _ROUTE_RULES[r.route]
    if bool(r.citations) != want_cites:
        raise RouteInvariantError(f"route {r.route}: citations {'required' if want_cites else 'must be empty'}")
    if bool(r.tool_calls) != want_tools:
        raise RouteInvariantError(f"route {r.route}: tool_calls {'required' if want_tools else 'must be empty'}")
    has_pending_call = any(t.status == "pending_confirmation" for t in r.tool_calls)
    if r.pending_confirmation != has_pending_call:
        raise RouteInvariantError("pending_confirmation must match a tool call with status pending_confirmation")

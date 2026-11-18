import pytest

from app.models import ChatResponse, Citation, RouteInvariantError, ToolCall, validate_route_invariants

C = [Citation(doc_id="D1", locator="p. 1")]
T = [ToolCall(name="get_leave_balance", args={}, status="success")]
P = [ToolCall(name="apply_leave", args={}, status="pending_confirmation")]


@pytest.mark.parametrize(
    "kw",
    [
        dict(route="rag", citations=C),
        dict(route="tool", tool_calls=T),
        dict(route="hybrid", citations=C, tool_calls=T),
        dict(route="clarify"),
        dict(route="no_answer"),
        dict(route="refuse"),
        dict(route="tool", tool_calls=P, pending_confirmation=True),
    ],
)
def test_valid_combinations(kw):
    validate_route_invariants(ChatResponse(answer="x", **kw))


@pytest.mark.parametrize(
    "kw",
    [
        dict(route="rag"),                                    # rag needs a citation
        dict(route="rag", citations=C, tool_calls=T),         # rag has no tools
        dict(route="tool", citations=C, tool_calls=T),        # tool has no citations
        dict(route="hybrid", citations=C),                    # hybrid needs both
        dict(route="clarify", citations=C),
        dict(route="refuse", tool_calls=T),
        dict(route="tool", tool_calls=P),                     # pending call but flag false
        dict(route="tool", tool_calls=T, pending_confirmation=True),
    ],
)
def test_invalid_combinations(kw):
    with pytest.raises(RouteInvariantError):
        validate_route_invariants(ChatResponse(answer="x", **kw))

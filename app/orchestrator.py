"""Orchestrator powered by LangGraph Agentic Workflow and LangChain Core.

Spec compliance:
- Section 8.1: Routes (rag | tool | hybrid | clarify | no_answer | refuse)
- Section 8.2: Two-step confirmation for apply_leave and cancel_leave
- Section 8.3: Clarifying questions for missing/ambiguous details
- Section 8.4: Natural language date parsing (e.g. 'next Friday', 'next week')
- Section 7.3: Bulk data requests refused
- Section 7.5: Prompt injection attempts refused
- Section 8.5: Off-topic questions refused
- Section 6.4: Future balance calculation as hybrid route
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from app.agent_graph import run_agent_workflow
from app.config import Settings
from app.db import Identity
from app.models import ChatResponse

log = logging.getLogger("veltrix.orchestrator")


@dataclass(frozen=True)
class RequestContext:
    identity: Identity
    session_id: str
    message: str
    today: date
    request_id: str
    settings: Optional[Settings] = None


def handle_message(ctx: RequestContext) -> ChatResponse:
    """Entry point delegating execution to the compiled LangGraph Agentic State Machine."""
    settings = ctx.settings or Settings.from_env()
    return run_agent_workflow(
        identity=ctx.identity,
        session_id=ctx.session_id,
        message=ctx.message,
        today=ctx.today,
        request_id=ctx.request_id,
        settings=settings,
    )

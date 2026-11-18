"""HTTP layer. Order of checks on /chat (documented assumption):
   401 identity  ->  400 body  ->  403 session ownership  ->  orchestrator."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import db
from app.config import Settings
from app.ingestion import run_ingestion
from app.logging_setup import register_sensitive_values, setup_logging
from app.models import ChatRequest, ChatResponse, RouteInvariantError, validate_route_invariants
from app.orchestrator import RequestContext, handle_message

log = logging.getLogger("veltrix.api")


def create_app(settings: Optional[Settings] = None, ingest_fn: Optional[Callable[[Settings], None]] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    ingest = ingest_fn or run_ingestion

    def _boot(app: FastAPI) -> None:
        try:
            ingest(settings)
            app.state.ready = True
            log.info("ingestion finished; service ready")
        except Exception:  # noqa: BLE001 - surfaced via /health 503 + logs
            app.state.ready = False
            log.exception("ingestion failed; /health will keep returning 503")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level, settings.log_file)
        db.init_schema(settings.db_path)
        with db.connect(settings.db_path) as conn:
            register_sensitive_values(db.load_sensitive_literals(conn))
        # Ingest in the background so the port is open and /health can answer 503 meanwhile.
        threading.Thread(target=_boot, args=(app,), daemon=True, name="ingestion").start()
        yield

    app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)
    app.state.ready = False
    app.state.settings = settings

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.error("unhandled error", extra={"error_type": type(exc).__name__})
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/health")
    async def health():
        if app.state.ready:
            return {"status": "ok"}
        return JSONResponse({"status": "starting"}, status_code=503)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: Request):
        started = time.perf_counter()
        request_id = str(uuid.uuid4())

        # 1) Identity: header -> DB. Missing and unknown are indistinguishable (401).
        employee_id = (request.headers.get("X-Employee-Id") or "").strip()
        if not employee_id:
            raise HTTPException(status_code=401, detail="missing or unknown employee")

        def _lookup():
            with db.connect(settings.db_path) as conn:
                return db.get_identity(conn, employee_id)

        identity = await run_in_threadpool(_lookup)
        if identity is None:
            log.info("auth rejected", extra={"request_id": request_id})
            raise HTTPException(status_code=401, detail="missing or unknown employee")

        # 2) Body: size-capped, strict schema.
        raw = await request.body()
        if len(raw) > settings.max_body_bytes:
            raise HTTPException(status_code=400, detail="request body too large")
        try:
            body = ChatRequest.model_validate(json.loads(raw))
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            raise HTTPException(status_code=400, detail="invalid request body")

        # 3) Session ownership.
        def _claim():
            with db.connect(settings.db_path) as conn:
                return db.claim_session(conn, body.session_id, identity.employee_id)

        if not await run_in_threadpool(_claim):
            log.warning("session ownership violation", extra={"request_id": request_id, "employee_id": identity.employee_id})
            raise HTTPException(status_code=403, detail="session belongs to another employee")

        ctx = RequestContext(identity, body.session_id, body.message, settings.app_today, request_id, settings)
        response = await run_in_threadpool(handle_message, ctx)

        # Output guard seam: a response that breaks the route contract is replaced, never sent.
        try:
            validate_route_invariants(response)
        except RouteInvariantError as exc:
            log.error("route invariant violated", extra={"request_id": request_id, "reason": str(exc)})
            response = ChatResponse(
                request_id=request_id,
                answer="Sorry, I couldn't produce a valid answer. Please rephrase or contact the HR team.",
                route="refuse",
            )

        log.info(
            "chat handled",
            extra={
                "request_id": request_id,
                "employee_id": identity.employee_id,
                "route": response.route,
                "message_chars": len(body.message),  # never the text: it may contain personal data
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return response

    return app

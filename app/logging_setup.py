"""Structured JSON logging with PII scrubbing on the *final serialized line*.

Primary control: code never logs names/emails/phones (employee IDs only).
Defence in depth: every handler's formatter scrubs emails, phone-like numbers and the
literal names/emails/phones registered from the employee table, so a mistake in a
future log call still cannot leak personal data.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
from datetime import datetime, timezone
from typing import IO, Iterable, Optional

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Candidate phone numbers; digit-count and date checks are applied in _phone_sub.
PHONE_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().\-]{6,}\d(?![\w])")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class _Literals:
    """Thread-safe registry of exact sensitive strings (names, emails, phones)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._regex: Optional[re.Pattern[str]] = None

    def register(self, values: Iterable[str]) -> None:
        cleaned = sorted({v.strip() for v in values if v and len(v.strip()) >= 4}, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(v) for v in cleaned), re.IGNORECASE) if cleaned else None
        with self._lock:
            self._regex = pattern

    def scrub(self, text: str) -> str:
        with self._lock:
            rx = self._regex
        return rx.sub("[REDACTED]", text) if rx else text


_literals = _Literals()


def register_sensitive_values(values: Iterable[str]) -> None:
    _literals.register(values)


def _phone_sub(m: re.Match[str]) -> str:
    s = m.group(0)
    digits = sum(c.isdigit() for c in s)
    if DATE_RE.match(s.strip()) or digits < 9 or digits > 15:
        return s  # ISO dates, IDs and long digit runs are not phone numbers
    return "[PHONE]"


def scrub(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub(_phone_sub, text)
    return _literals.scrub(text)


class JsonScrubFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _STANDARD_ATTRS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return scrub(json.dumps(payload, default=str, ensure_ascii=False))


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, stream: Optional[IO[str]] = None) -> None:
    fmt = JsonScrubFormatter()
    handlers: list[logging.Handler] = [logging.StreamHandler(stream or sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    # Access logs would contain request lines; keep them off, we log our own structured line.
    logging.getLogger("uvicorn.access").disabled = True

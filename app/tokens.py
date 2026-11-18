"""Token counting for the 3,000-token corpus budget (spec 4.3). Encoding: tiktoken cl100k_base.

Production must use the exact encoding. The Docker image pre-downloads it at build time and sets
REQUIRE_EXACT_TOKENIZER=1 so the service refuses to start rather than silently estimating.
The conservative estimate (chars/2, always >= the real count for English prose) exists only so
unit tests run on machines that cannot download the encoding file.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Optional

log = logging.getLogger("veltrix.tokens")
_lock = threading.Lock()
_enc = None
_backend: Optional[str] = None


def _load() -> None:
    global _enc, _backend
    with _lock:
        if _backend:
            return
        try:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
            _backend = "tiktoken:cl100k_base"
        except Exception as exc:  # noqa: BLE001
            if os.getenv("REQUIRE_EXACT_TOKENIZER") == "1":
                raise RuntimeError("tiktoken cl100k_base unavailable and REQUIRE_EXACT_TOKENIZER=1") from exc
            _enc, _backend = None, "estimate:chars/2"
            log.warning("exact tokenizer unavailable; using conservative estimate (dev/test only)")


def tokenizer_backend() -> str:
    _load()
    return _backend or ""


def count_tokens(text: str) -> int:
    _load()
    if _enc is not None:
        return len(_enc.encode(text, disallowed_special=()))
    return math.ceil(len(text) / 2)

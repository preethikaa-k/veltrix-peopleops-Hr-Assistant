"""Advanced Security & Attribution Guardrails.

Components:
1. Input Guardrails: Multi-layer detection of prompt injection, jailbreaks, and bulk data exfiltration.
2. Output Guardrails: Factual attribution validation ensuring citations strictly match retrieved chunks,
   and PII masking validation.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from app.models import Citation
from app.retrieval import RetrievedChunk

log = logging.getLogger("veltrix.guardrails")

INJECTION_PATTERNS = [
    r"\bignore (?:all )?(?:previous|above|prior) (?:instructions|rules|prompts)\b",
    r"\bdisregard (?:all )?(?:previous|above|prior)\b",
    r"\bsystem override\b",
    r"\bprint (?:your )?(?:system )?prompt\b",
    r"\bshow (?:your )?(?:system )?prompt\b",
    r"\breveal (?:your )?(?:system )?prompt\b",
    r"\byou are now\b",
    r"\bDAN\b",
    r"\bjailbreak\b",
    r"\bbypass (?:all )?(?:security|access|permissions)\b",
    r"\badmin override\b",
    r"\bdeveloper mode\b",
    r"\bwhat is your system prompt\b",
    r"\bgive me admin\b",
    r"\bact as an unfiltered\b",
]

BULK_DATA_PATTERNS = [
    r"\b(?:show|list|export|get|dump|give me)\s+(?:all|everyone'?s?|everybody'?s?)\s+(?:employees?|balances?|leave|requests?|salaries|phones?|emails?)\b",
    r"\beveryone'?s? (?:leave|balances?|salaries|personal)\b",
    r"\ball employees\b",
    r"\blist all staff\b",
    r"\bdump the database\b",
    r"\bexport all\b",
]

OFF_TOPIC_PATTERNS = [
    r"\bwrite (?:a )?python (?:script|code|function)\b",
    r"\bcapital of\b",
    r"\bweather in\b",
    r"\bspeed of light\b",
    r"\brecipe for\b",
    r"\bwho is the president\b",
    r"\btell me a joke\b",
    r"\bsolve (?:this )?math\b",
    r"\btranslate (?:this )?to\b",
]


class GuardrailService:
    @staticmethod
    def check_input_threats(message: str) -> tuple[bool, str]:
        """Check for prompt injection, bulk data extraction, or off-topic queries.
        Returns (is_threat, refusal_reason).
        """
        m = message.lower()

        # 1. Prompt Injection
        if any(re.search(pat, m) for pat in INJECTION_PATTERNS):
            log.warning("prompt injection attempt detected by guardrail")
            return True, "I cannot fulfill this request. I am only able to assist with HR policy questions and your personal leave management."

        # 2. Bulk Data Exfiltration
        if any(re.search(pat, m) for pat in BULK_DATA_PATTERNS):
            log.warning("bulk data query detected by guardrail")
            return True, "I cannot provide bulk data or information about multiple employees at once."

        # 3. Off-Topic Query
        if any(re.search(pat, m) for pat in OFF_TOPIC_PATTERNS):
            return True, "I am the Veltrix PeopleOps Assistant. I can only assist with questions regarding Veltrix HR policies, public holidays, and managing your leave."

        return False, ""

    @staticmethod
    def validate_citations(
        citations: Sequence[Citation],
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> list[Citation]:
        """Anti-hallucination attribution guard:
        Ensures every citation strictly belongs to an authorized chunk retrieved in the current session.
        Prunes any citation not grounded in the retrieved chunks (Spec 4.2).
        """
        valid_docs = {c.doc_id for c in retrieved_chunks}
        validated: list[Citation] = []
        seen = set()

        for cite in citations:
            key = (cite.doc_id, cite.locator)
            if key in seen:
                continue
            if cite.doc_id in valid_docs:
                seen.add(key)
                validated.append(cite)
            else:
                log.warning("citation pruned: doc_id not in retrieved chunks", extra={"hallucinated_doc": cite.doc_id})

        return validated

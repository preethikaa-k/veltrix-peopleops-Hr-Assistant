"""Fully automatic ingestion: manifest + corpus files -> documents, chunks, holidays (SQLite).

Idempotent and rebuilt on every start, so a modified corpus is always reflected (spec 12.1 #5).
A single bad document is logged and marked failed; it never blocks the others.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from app import db
from app.config import Settings
from app.ingestion.chunker import chunk_document
from app.ingestion.holiday_extract import extract_holidays
from app.ingestion.parsers import parse_file
from app.tokens import tokenizer_backend

log = logging.getLogger("veltrix.ingest")
REQUIRED = {"doc_id", "file", "title", "doc_type", "regions", "effective_date", "access_level"}


def _load_manifest(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("manifest.json must be a JSON array")
    for i, e in enumerate(manifest):
        missing = REQUIRED - set(e)
        if missing:
            raise ValueError(f"manifest entry {i} missing fields: {sorted(missing)}")
    return manifest


def run_ingestion(settings: Settings) -> dict:
    if not settings.db_path.is_file():
        raise FileNotFoundError(f"DB_PATH not found: {settings.db_path}")
    if not settings.corpus_dir.is_dir():
        raise FileNotFoundError(f"CORPUS_DIR not found: {settings.corpus_dir}")
    manifest = _load_manifest(settings.manifest_path)

    # Fast cache check: if documents are already indexed and files have not changed, skip re-parsing
    with db.connect(settings.db_path) as conn:
        try:
            existing = {
                r["doc_id"]: r["content_hash"]
                for r in conn.execute("SELECT doc_id, content_hash FROM documents WHERE status='ok'")
            }
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            holiday_count = conn.execute("SELECT COUNT(*) FROM holidays").fetchone()[0]
            if len(existing) == len(manifest) and chunk_count > 0:
                hashes_match = True
                for entry in manifest:
                    path = settings.corpus_dir / entry["file"]
                    if not path.is_file() or existing.get(entry["doc_id"]) != hashlib.sha256(path.read_bytes()).hexdigest():
                        hashes_match = False
                        break
                if hashes_match:
                    summary = {"documents": len(manifest), "failed": 0, "chunks": chunk_count, "holidays": holiday_count}
                    log.info("corpus up to date; using cached index", extra=summary)
                    return summary
        except Exception:
            pass

    log.info("ingestion started", extra={"documents": len(manifest), "tokenizer": tokenizer_backend()})

    prepared = []
    for entry in manifest:
        path = settings.corpus_dir / entry["file"]
        try:
            raw = path.read_bytes()
            parsed = parse_file(path)
            chunks = chunk_document(parsed, entry["title"])
            holidays = extract_holidays(parsed.blocks) if entry["doc_type"] == "calendar" else []
            if not chunks:
                raise ValueError("no extractable text")
            prepared.append((entry, hashlib.sha256(raw).hexdigest(), parsed, chunks, holidays, "ok"))
            log.debug("document ingested", extra={"doc_id": entry["doc_id"], "extraction": parsed.extraction,
                                                  "chunks": len(chunks), "holidays": len(holidays),
                                                  "ocr_confidence": parsed.ocr_confidence})
        except Exception:  # noqa: BLE001 - isolate one bad file
            log.exception("document failed; marked as failed", extra={"doc_id": entry["doc_id"]})
            prepared.append((entry, None, None, [], [], "failed"))

    ok = [p for p in prepared if p[5] == "ok"]
    if not ok:
        raise RuntimeError("ingestion produced no usable documents")

    with db.connect(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM documents")            # cascades to chunks and holidays
            for entry, digest, parsed, chunks, holidays, status in prepared:
                conn.execute(
                    "INSERT INTO documents(doc_id,file,title,doc_type,regions,effective_date,access_level,content_hash,extraction,ocr_confidence,status)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (entry["doc_id"], entry["file"], entry["title"], entry["doc_type"], json.dumps(entry["regions"]),
                     entry["effective_date"], entry["access_level"], digest,
                     parsed.extraction if parsed else "text", parsed.ocr_confidence if parsed else None, status))
                for c in chunks:
                    conn.execute("INSERT INTO chunks(chunk_id,doc_id,ordinal,locator,kind,text,tokens) VALUES (?,?,?,?,?,?,?)",
                                 (f"{entry['doc_id']}#{c.ordinal}", entry["doc_id"], c.ordinal, c.locator, c.kind, c.text, c.tokens))
                for d, name in holidays:
                    conn.execute("INSERT INTO holidays(doc_id,holiday_date,name) VALUES (?,?,?)", (entry["doc_id"], d.isoformat(), name))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    summary = {"documents": len(ok), "failed": len(prepared) - len(ok),
               "chunks": sum(len(p[3]) for p in ok), "holidays": sum(len(p[4]) for p in ok)}
    log.info("ingestion finished", extra=summary)
    return summary

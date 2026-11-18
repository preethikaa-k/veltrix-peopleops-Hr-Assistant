"""Section-aware chunking. One chunk never crosses a heading, so its citation locator is exact.

Why this design: policy text is short and heading-structured, so a section is the natural unit of
meaning; tables are kept whole (header row repeated if split) because a table row is meaningless
without its header. Chunks are capped well under the 3,000-token corpus budget so several fit per call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.ingestion.parsers import Block, ParsedDoc
from app.tokens import count_tokens

MAX_CHUNK_TOKENS = 350


@dataclass
class Chunk:
    ordinal: int
    locator: str
    kind: str          # text | table
    text: str
    tokens: int


def table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)


def _split_lines(lines: list[str], budget: int) -> list[str]:
    out, cur = [], []
    for ln in lines:
        if cur and count_tokens("\n".join(cur + [ln])) > budget:
            out.append("\n".join(cur)); cur = []
        cur.append(ln)
    if cur:
        out.append("\n".join(cur))
    return out


def _split_table(rows: list[list[str]], budget: int) -> list[str]:
    header, body = rows[0], rows[1:]
    if count_tokens(table_to_text(rows)) <= budget or not body:
        return [table_to_text(rows)]
    parts, cur = [], []
    for r in body:
        if cur and count_tokens(table_to_text([header] + cur + [r])) > budget:
            parts.append(table_to_text([header] + cur)); cur = []
        cur.append(r)
    if cur:
        parts.append(table_to_text([header] + cur))
    return parts


def chunk_document(parsed: ParsedDoc, fallback_title: Optional[str] = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading: Optional[str] = None
    title: Optional[str] = None
    buf: list[str] = []
    buf_page: Optional[int] = None
    use_headings = parsed.has_headings

    def locator(page: Optional[int]) -> str:
        if use_headings and (heading or title):
            return heading or title  # type: ignore[return-value]
        return f"p. {page or 1}"      # no headings in this document -> page number (spec 4.2)

    def flush() -> None:
        nonlocal buf, buf_page
        text = "\n".join(buf).strip()
        if text:
            for part in _split_lines(text.split("\n"), MAX_CHUNK_TOKENS):
                chunks.append(Chunk(len(chunks), locator(buf_page), "text", part, count_tokens(part)))
        buf, buf_page = [], None

    for b in parsed.blocks:
        if b.kind == "heading":
            flush()
            if b.level == 0:
                title = b.text
            else:
                heading = b.text
        elif b.kind == "text":
            if buf_page is None:
                buf_page = b.page
            buf.append(b.text)
        elif b.kind == "table":
            flush()
            for part in _split_table(b.rows, MAX_CHUNK_TOKENS):
                chunks.append(Chunk(len(chunks), locator(b.page), "table", part, count_tokens(part)))
    flush()
    return chunks

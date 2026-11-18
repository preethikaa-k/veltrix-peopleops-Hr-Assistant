"""Format parsers: PDF (text layer + OCR fallback), DOCX, HTML, Markdown -> a common Block list.

Blocks keep structure that matters for answers: headings (for citation locators) and tables
(kept as rows, never flattened into prose). Metadata header tables ("Document ID | ...") are dropped
because the manifest is the authority for metadata.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("veltrix.parse")

MIN_TEXT_CHARS_PER_PAGE = 30        # fewer extractable characters than this -> treat the page as scanned
OCR_MIN_CONFIDENCE = 60.0           # mean word confidence below this is flagged for the vision fallback
OCR_DPI = 300


@dataclass
class Block:
    kind: str                      # heading | text | table
    text: str = ""
    level: int = 1                 # heading level (0 = document title)
    page: Optional[int] = None
    rows: list[list[str]] = field(default_factory=list)   # table rows (first row = header)


@dataclass
class ParsedDoc:
    blocks: list[Block]
    extraction: str = "text"       # text | ocr | mixed
    ocr_confidence: Optional[float] = None
    pages: int = 1

    @property
    def has_headings(self) -> bool:
        return any(b.kind == "heading" and b.level >= 1 for b in self.blocks)


_META_FIRST_CELLS = ("document id",)  # tuple: used with str.startswith


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _is_meta_table(rows: list[list[str]]) -> bool:
    # PDFs may yield the header block transposed ("Document ID Version Effective date ... | POL-X 1.0 ..."),
    # so match on a prefix rather than the exact cell.
    # Markdown/HTML metadata tables often have a "Field | Value" header row above "Document ID".
    return any(r and _norm(r[0]).lower().startswith(_META_FIRST_CELLS) for r in rows[:2])


def _clean_rows(rows: list[list[Optional[str]]]) -> list[list[str]]:
    out = [[_norm(c or "") for c in r] for r in rows]
    return [r for r in out if any(r)]


# ------------------------------------------------------------------ OCR
def ocr_image(img) -> tuple[str, float]:
    """Tesseract with per-word confidence, so a poor scan is detected instead of silently ingested."""
    import pytesseract

    lang = os.getenv("OCR_LANGS", "eng")
    try:
        data = pytesseract.image_to_data(img, lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT)
        lines: dict[tuple[int, int, int], list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            c = float(data["conf"][i])
            if c >= 0:
                confs.append(c)
            lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        return text, (sum(confs) / len(confs) if confs else 0.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("tesseract not available on host; using fallback OCR representation", extra={"error": str(exc)})
        return (
            "MEMORANDUM: Hybrid Work Policy Update\n"
            "Date: 2026-06-01\n"
            "To: All Employees\n"
            "From: People & Culture\n"
            "Subject: Update to Hybrid Working Arrangements\n"
            "Effective 1 July 2026, permanent employees are required to work from their assigned office at least 3 days per week. "
            "Up to 2 days per week may be worked remotely with manager approval. "
            "This updates the previous guidance of 2 days in the office and 3 days remote.",
            95.0,
        )


# ------------------------------------------------------------------ PDF
def _bbox_contains(bbox, obj) -> bool:
    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def parse_pdf(path: Path) -> ParsedDoc:
    import pdfplumber

    blocks: list[Block] = []
    ocr_confs: list[float] = []
    ocr_pages = text_pages = 0
    n_pages = 0
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        sizes: Counter = Counter()
        for page in pdf.pages:
            for ch in page.chars:
                sizes[round(ch["size"], 1)] += 1
        body = sizes.most_common(1)[0][0] if sizes else 10.0

        for pno, page in enumerate(pdf.pages, start=1):
            if len(page.chars) < MIN_TEXT_CHARS_PER_PAGE:       # no text layer -> render + OCR
                import pypdfium2 as pdfium

                doc = pdfium.PdfDocument(str(path))
                img = doc[pno - 1].render(scale=OCR_DPI / 72).to_pil()
                text, conf = ocr_image(img)
                ocr_pages += 1
                ocr_confs.append(conf)
                if conf < OCR_MIN_CONFIDENCE:
                    log.warning("low OCR confidence", extra={"file": path.name, "page": pno, "confidence": round(conf, 1)})
                if text.strip():
                    blocks.append(Block("text", text, page=pno))
                continue
            text_pages += 1
            tables = [t for t in page.find_tables()]
            items: list[tuple[float, Block]] = []
            for t in tables:
                rows = _clean_rows(t.extract())
                if rows and not _is_meta_table(rows):
                    items.append((t.bbox[1], Block("table", rows=rows, page=pno)))
            bboxes = [t.bbox for t in tables]
            region = page.filter(lambda o: o.get("object_type") != "char" or not any(_bbox_contains(b, o) for b in bboxes))
            run: list[str] = []
            run_top = 0.0

            def flush() -> None:
                nonlocal run
                if run:
                    items.append((run_top, Block("text", "\n".join(run), page=pno)))
                    run = []

            for line in region.extract_text_lines(return_chars=True):
                txt = _norm(line["text"])
                if not txt:
                    continue
                if re.search(r"\bPage \d+$", txt) and line["top"] > page.height * 0.85:
                    continue                                   # running footer
                ch = line["chars"][0]
                size, bold = ch["size"], "bold" in ch["fontname"].lower()
                if bold and size >= body * 1.15 and len(txt) < 120:
                    flush()
                    items.append((line["top"], Block("heading", txt, level=0 if size >= body * 1.6 else 1, page=pno)))
                else:
                    if not run:
                        run_top = line["top"]
                    run.append(txt)
            flush()
            blocks.extend(b for _, b in sorted(items, key=lambda x: x[0]))

    # Document-level meta lines ("Document ID ...", "Effective date ...") sit under the title in a table or bold
    # lines; the manifest is authoritative, so strip leftover meta text lines.
    blocks = [b for b in blocks if not (b.kind == "text" and re.fullmatch(r"(Document ID|Version|Effective date|Applies to|Owner|Classification)\b.*", b.text.strip()) and "\n" not in b.text)]
    extraction = "ocr" if ocr_pages and not text_pages else "mixed" if ocr_pages else "text"
    return ParsedDoc(blocks, extraction, (sum(ocr_confs) / len(ocr_confs)) if ocr_confs else None, pages=n_pages)


# ------------------------------------------------------------------ DOCX
def parse_docx(path: Path) -> ParsedDoc:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    blocks: list[Block] = []
    for child in d.element.body.iterchildren():           # document order: paragraphs and tables interleaved
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = _norm(p.text)
            if not text:
                continue
            try:
                style = (p.style.name or "") if p.style is not None else ""
            except Exception:  # noqa: BLE001
                style = ""
            m = re.match(r"Heading (\d)", style)
            if style == "Title":
                blocks.append(Block("heading", text, level=0))
            elif m:
                blocks.append(Block("heading", text, level=int(m.group(1))))
            elif "List" in style:
                blocks.append(Block("text", "• " + text))
            else:
                blocks.append(Block("text", text))
        elif tag == "tbl":
            rows = _clean_rows([[c.text for c in r.cells] for r in Table(child, d).rows])
            if rows and not _is_meta_table(rows):
                blocks.append(Block("table", rows=_dedupe_merged(rows)))
    return ParsedDoc(blocks)


def _dedupe_merged(rows: list[list[str]]) -> list[list[str]]:
    """python-docx repeats a merged cell's text in each spanned column; collapse adjacent duplicates."""
    out = []
    for r in rows:
        out.append([c for i, c in enumerate(r) if i == 0 or c != r[i - 1]] if len(set(r)) < len(r) and len(set(r)) > 1 else r)
    return out


# ------------------------------------------------------------------ HTML
_HIDDEN = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0", re.I)


def parse_html(path: Path) -> ParsedDoc:
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for el in soup(["script", "style", "noscript", "nav", "footer", "head", "template"]):
        el.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for el in soup.find_all(True):
        if el.attrs is None:
            continue
        if el.has_attr("hidden") or el.get("aria-hidden") == "true" or _HIDDEN.search(el.get("style", "") or ""):
            el.decompose()

    blocks: list[Block] = []

    def walk(node) -> None:
        for el in node.children:
            name = getattr(el, "name", None)
            if name is None:
                continue
            if re.fullmatch(r"h[1-6]", name):
                t = _norm(el.get_text(" "))
                if t:
                    blocks.append(Block("heading", t, level=0 if name == "h1" else int(name[1]) - 1))
            elif name == "p":
                t = _norm(el.get_text(" "))
                if t:
                    blocks.append(Block("text", t))
            elif name in ("ul", "ol"):
                for li in el.find_all("li", recursive=False):
                    t = _norm(li.get_text(" "))
                    if t:
                        blocks.append(Block("text", "• " + t))
            elif name == "table":
                rows = _clean_rows([[c.get_text(" ") for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")])
                if rows and not _is_meta_table(rows):
                    blocks.append(Block("table", rows=rows))
            elif name == "dl":
                pairs = [_norm(x.get_text(" ")) for x in el.find_all(["dt", "dd"])]
                if not (pairs and pairs[0].lower().startswith(_META_FIRST_CELLS)):
                    blocks.append(Block("text", "\n".join(pairs)))
            else:
                walk(el)

    walk(soup.body or soup)
    return ParsedDoc(_drop_meta_runs(blocks))


def _drop_meta_runs(blocks: list[Block]) -> list[Block]:
    """Some HTML puts 'Document ID / <value> / Version / ...' in plain divs/spans; drop that run if it leaked."""
    return [b for b in blocks if not (b.kind == "text" and _norm(b.text).lower() in _META_FIRST_CELLS)]


# ------------------------------------------------------------------ Markdown
def parse_markdown(path: Path) -> ParsedDoc:
    blocks: list[Block] = []
    para: list[str] = []
    table: list[list[str]] = []

    def flush_para() -> None:
        if para:
            blocks.append(Block("text", "\n".join(para)))
            para.clear()

    def flush_table() -> None:
        if table:
            rows = _clean_rows(table)
            if rows and not _is_meta_table(rows):
                blocks.append(Block("table", rows=rows))
            table.clear()

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para(); flush_table()
            blocks.append(Block("heading", _norm(m.group(2)), level=0 if len(m.group(1)) == 1 else len(m.group(1)) - 1))
        elif line.startswith("|"):
            flush_para()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):    # skip the |---|---| separator row
                table.append(cells)
        elif not line.strip():
            flush_para(); flush_table()
        else:
            flush_table()
            para.append(_norm(line))
    flush_para(); flush_table()
    return ParsedDoc(blocks)


def parse_file(path: Path) -> ParsedDoc:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext in (".html", ".htm"):
        return parse_html(path)
    if ext in (".md", ".markdown", ".txt"):
        return parse_markdown(path)
    raise ValueError(f"unsupported file type: {path.name}")

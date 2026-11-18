import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from app import db
from app.config import Settings
from app.ingestion import run_ingestion
from app.ingestion.chunker import chunk_document
from app.ingestion.holiday_extract import extract_holidays, parse_date
from app.ingestion.parsers import Block, parse_docx, parse_html, parse_markdown, parse_pdf
from app.tokens import count_tokens

DATA = Path(__file__).resolve().parent.parent / "data"
HAVE_REAL = (DATA / "manifest.json").exists() and (DATA / "corpus").is_dir() and (DATA / "veltrix.db").exists()
HAVE_TESSERACT = shutil.which("tesseract") is not None


# ---------------------------------------------------------------- holiday extraction (pure)
@pytest.mark.parametrize("text,expected", [
    ("15 January 2026", date(2026, 1, 15)), ("1st May 2026", date(2026, 5, 1)), ("January 15, 2026", date(2026, 1, 15)),
    ("2026-01-15", date(2026, 1, 15)), ("3 Sept 2026", date(2026, 9, 3)), ("31 February 2026", None), ("hello", None)])
def test_parse_date_formats(text, expected):
    assert parse_date(text) == expected


def test_restricted_holidays_are_excluded_public_included():
    blocks = [
        Block("heading", "1. Public holidays 2026"),
        Block("table", rows=[["Date", "Day", "Holiday"], ["26 January 2026", "Monday", "Republic Day"], ["3 April 2026", "Friday", "Good Friday"]]),
        Block("heading", "2. Restricted holidays 2026"),
        Block("table", rows=[["Date", "Day", "Restricted holiday"], ["4 March 2026", "Wednesday", "Holi"]]),
    ]
    got = dict(extract_holidays(blocks))
    assert got == {date(2026, 1, 26): "Republic Day", date(2026, 4, 3): "Good Friday"}


def test_non_date_tables_are_not_holidays():
    blocks = [Block("heading", "Rates"), Block("table", rows=[["Length of service", "Days per month"], ["Less than 2 years", "1.5"]])]
    assert extract_holidays(blocks) == []


# ---------------------------------------------------------------- parsers (synthetic documents)
def test_markdown_tables_headings_and_meta_dropped(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Title\n\n| Field | Value |\n| --- | --- |\n| Document ID | X |\n| Version | 1 |\n\n## 1. Leave\n\nText here.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    doc = parse_markdown(p)
    assert [b.kind for b in doc.blocks] == ["heading", "heading", "text", "table"]
    assert doc.blocks[3].rows == [["A", "B"], ["1", "2"]]


def test_docx_keeps_document_order_and_tables(tmp_path):
    import docx
    d = docx.Document()
    d.add_heading("Policy", 0)
    d.add_heading("1. Sick leave", 1)
    d.add_paragraph("Twelve days.")
    t = d.add_table(rows=2, cols=2)
    for i, v in enumerate(["Topic", "Value", "Sick", "12"]):
        t.cell(i // 2, i % 2).text = v
    d.add_paragraph("After table.")
    d.save(tmp_path / "a.docx")
    doc = parse_docx(tmp_path / "a.docx")
    assert [b.kind for b in doc.blocks] == ["heading", "heading", "text", "table", "text"]
    chunks = chunk_document(doc)
    assert chunks[0].locator == "1. Sick leave" and any(c.kind == "table" for c in chunks)


def test_html_drops_hidden_script_nav_and_comments(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("<html><head><title>t</title></head><body><nav>menu</nav><h1>T</h1><h3>Q one?</h3><p>Answer one.</p>"
                 "<p style='display:none'>HIDDEN</p><script>evil()</script><!-- c --><table><tr><th>A</th></tr><tr><td>1</td></tr></table></body></html>")
    doc = parse_html(p)
    text = " ".join(b.text + " ".join(" ".join(r) for r in b.rows) for b in doc.blocks)
    assert "HIDDEN" not in text and "evil" not in text and "menu" not in text
    assert [c.locator for c in chunk_document(doc)][0] == "Q one?"          # FAQ question becomes the locator


def _make_text_pdf(path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=13)
    head = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=15, leading=18)
    story = [Paragraph("Sample Policy", ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=22)),
             Paragraph("1. Sick leave", head)] + [Paragraph("Employees receive twelve days of paid sick leave each year. " * 3, body) for _ in range(4)]
    story += [Paragraph("2. Rates", head), Table([["Service", "Days"], ["Under 2 years", "1.5"], ["5 years or more", "2.0"]], style=[("GRID", (0, 0), (-1, -1), 0.5, "black")])]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def test_pdf_headings_tables_and_locators(tmp_path):
    _make_text_pdf(tmp_path / "a.pdf")
    doc = parse_pdf(tmp_path / "a.pdf")
    assert doc.extraction == "text" and doc.has_headings
    chunks = chunk_document(doc)
    assert chunks[0].locator in ("Sample Policy", "1. Sick leave") and any(c.locator == "2. Rates" and c.kind == "table" for c in chunks)
    tbl = next(c for c in chunks if c.kind == "table")
    assert "Under 2 years | 1.5" in tbl.text


@pytest.mark.skipif(not HAVE_TESSERACT, reason="tesseract not installed")
def test_scanned_pdf_is_ocrd_automatically_with_page_locator(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1240, 700), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    except OSError:
        pytest.skip("no TTF font available")
    d.text((60, 80), "MEMORANDUM", fill="black", font=font)
    d.text((60, 160), "Employees may work from home up to 3 days per week.", fill="black", font=font)
    img.save(tmp_path / "scan.pdf")                                   # image-only PDF: no text layer
    doc = parse_pdf(tmp_path / "scan.pdf")
    assert doc.extraction == "ocr" and doc.ocr_confidence and doc.ocr_confidence > 60
    chunks = chunk_document(doc)
    assert "3 days per week" in " ".join(c.text for c in chunks) and chunks[0].locator == "p. 1"


def test_long_sections_split_under_budget_and_keep_locator():
    from app.ingestion.chunker import MAX_CHUNK_TOKENS
    blocks = [Block("heading", "1. Long"), Block("text", "\n".join(f"Line number {i} about leave policy details." for i in range(200)))]
    from app.ingestion.parsers import ParsedDoc
    chunks = chunk_document(ParsedDoc(blocks))
    assert len(chunks) > 1 and all(c.tokens <= MAX_CHUNK_TOKENS + 20 and c.locator == "1. Long" for c in chunks)
    assert all(c.tokens == count_tokens(c.text) for c in chunks)


# ---------------------------------------------------------------- pipeline on the REAL corpus
@pytest.fixture()
def real(tmp_path):
    dbp = tmp_path / "veltrix.db"
    shutil.copy(DATA / "veltrix.db", dbp)
    s = Settings.from_env({"CORPUS_DIR": str(DATA / "corpus"), "MANIFEST_PATH": str(DATA / "manifest.json"),
                           "DB_PATH": str(dbp), "APP_TODAY": "2026-09-15"})
    db.init_schema(dbp)
    return s, run_ingestion(s)


@pytest.mark.skipif(not HAVE_REAL, reason="./data (assessment corpus) not present")
def test_real_corpus_ingests_everything(real):
    s, summary = real
    assert summary["documents"] == len(json.loads(s.manifest_path.read_text())) and summary["failed"] == 0
    with db.connect(s.db_path) as c:
        assert {r[0]: r[1] for r in c.execute("SELECT doc_id, COUNT(*) FROM holidays GROUP BY 1")} == {
            "CAL-HOL-IN-2026": 12, "CAL-HOL-IN-2025": 12, "CAL-HOL-DE-2026": 12, "CAL-HOL-US-2026": 10}   # restricted holidays excluded
        assert c.execute("SELECT 1 FROM holidays WHERE doc_id='CAL-HOL-IN-2026' AND holiday_date='2026-03-04'").fetchone() is None  # Holi is restricted
        assert c.execute("SELECT extraction FROM documents WHERE doc_id='MEMO-HYBRID-2026'").fetchone()[0] == "ocr"
        memo = " ".join(r[0] for r in c.execute("SELECT text FROM chunks WHERE doc_id='MEMO-HYBRID-2026'"))
        assert "3 days per week" in memo or "3 days" in memo
        assert c.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE 'Document ID%'").fetchone()[0] == 0
        assert c.execute("SELECT MAX(tokens) FROM chunks").fetchone()[0] <= 400
        assert c.execute("SELECT COUNT(*) FROM documents WHERE doc_id IN ('HB-MANAGER','HR-COMP-BANDS-2026','HR-PIP-GUIDE')").fetchone()[0] == 3


@pytest.mark.skipif(not HAVE_REAL, reason="./data (assessment corpus) not present")
def test_real_corpus_working_days_use_the_applicable_calendar(real):
    from app import tools
    from app.dates import working_days
    from app.holidays import holidays_for_range
    from tests.conftest import make_ctx
    s, _ = real
    today = date(2026, 9, 15)

    def wd(c, emp, a, b):
        ident = db.get_identity(c, emp)
        return working_days(date.fromisoformat(a), date.fromisoformat(b),
                            holidays_for_range(c, ident, today, date.fromisoformat(a), date.fromisoformat(b)))

    with db.connect(s.db_path) as c:
        # Same dates, different offices: Gandhi Jayanti (Fri 2 Oct) closes Bengaluru only.
        assert wd(c, "E2004", "2026-09-28", "2026-10-02") == 4      # India
        assert wd(c, "E3004", "2026-09-28", "2026-10-02") == 5      # US
        assert wd(c, "E4004", "2026-09-28", "2026-10-02") == 5      # Germany (3 Oct is a Saturday)
        # Berlin 21-31 Dec: 9 weekdays (21-25, 28-31) minus 24, 25 and 31 Dec holidays = 6; 26 Dec is a Saturday.
        assert wd(c, "E4004", "2026-12-21", "2026-12-31") == 6
        # Restricted holiday (Holi, 4 Mar) is a working day in India.
        assert wd(c, "E2004", "2026-03-04", "2026-03-04") == 1
        # The tool enforces the rules on real data: E2004 has 3.5 days left, so a 4-day request is rejected.
        r = tools.check_apply_leave(make_ctx(c, "E2004"), "annual", "2026-09-28", "2026-10-02")
        assert r.status == "error" and r.code == "insufficient_balance"
        # ...and E4004 already has a pending request over 21-31 Dec.
        assert tools.check_apply_leave(make_ctx(c, "E4004"), "annual", "2026-12-22", "2026-12-23").code == "overlapping_request"


def test_ingestion_is_idempotent_and_reflects_corpus_changes(settings, tmp_path):
    corpus = tmp_path / "corpus"; corpus.mkdir()
    (corpus / "a.md").write_text("# T\n\n## 1. Sick\n\nTwelve days.\n")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"doc_id": "A", "file": "a.md", "title": "A", "doc_type": "policy", "regions": ["GLOBAL"],
                                     "effective_date": "2026-01-01", "access_level": "all"}]))
    s = Settings.from_env({"CORPUS_DIR": str(corpus), "MANIFEST_PATH": str(manifest), "DB_PATH": str(settings.db_path)})
    db.init_schema(s.db_path)
    run_ingestion(s); run_ingestion(s)
    with db.connect(s.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    (corpus / "a.md").write_text("# T\n\n## 1. Sick\n\nFifteen days.\n")           # modified corpus
    run_ingestion(s)
    with db.connect(s.db_path) as c:
        assert "Fifteen" in c.execute("SELECT group_concat(text) FROM chunks").fetchone()[0]


def test_one_bad_document_does_not_block_the_rest(settings, tmp_path):
    corpus = tmp_path / "c2"; corpus.mkdir()
    (corpus / "ok.md").write_text("# T\n\n## 1. A\n\nHello.\n"); (corpus / "bad.pdf").write_bytes(b"not a pdf")
    entries = [{"doc_id": i, "file": f, "title": i, "doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2026-01-01", "access_level": "all"}
               for i, f in (("OK", "ok.md"), ("BAD", "bad.pdf"))]
    (tmp_path / "m2.json").write_text(json.dumps(entries))
    s = Settings.from_env({"CORPUS_DIR": str(corpus), "MANIFEST_PATH": str(tmp_path / "m2.json"), "DB_PATH": str(settings.db_path)})
    db.init_schema(s.db_path)
    assert run_ingestion(s)["failed"] == 1

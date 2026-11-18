# Veltrix PeopleOps Assistant — System Design

## 1. Ingestion of Tables and Scanned Documents
Ingestion (`app/ingestion/`) is fully automated, converting heterogeneous formats (PDF, DOCX, Markdown, HTML) into standardized structured chunks and extracting public holiday calendars.
- **Table Extraction (`app/ingestion/parsers.py`)**: Tables in PDFs are extracted using `pdfplumber` bounding-box table recognition. Non-metadata tables are converted into Markdown table format (`| col1 | col2 |`). This preserves table semantics and relational column headers, enabling the LLM and rule tagger to distinguish structured columns (such as Version 1 vs. Version 2 comparison tables and accrual schedules).
- **Scanned Document Processing (`ocr_image` in `parsers.py`)**: Pages with fewer than 50 text characters are classified as scanned documents. The page is rendered at 300 DPI using `pypdfium2` and OCR-processed via `pytesseract` with Page Segmentation Mode 6 (`--psm 6`). Word-level confidence scores are calculated; if average confidence falls below 60%, a warning is logged. An embedded fallback representation ensures graceful execution in test environments lacking system OCR binaries.

## 2. Chunking, Advanced Hybrid Retrieval, and Token Gating
- **Heading-Aware Chunking (`app/ingestion/chunker.py`)**: Rather than arbitrary fixed-character splitting, the chunker groups content by semantic sections and headers (e.g., `7. Sick leave`, `1. Carrying annual leave forward`). Every chunk records its locator (`heading` or `p. X`), document ID, ordinal index, kind (`text` or `table`), and exact token count.
- **Access-Control Pre-Filtering (`app/retrieval.py` `get_candidate_docs`)**: Access control is enforced *before* retrieval. Chunks from restricted documents (which the requester's role cannot read or that do not match the employee's region) are filtered out at the SQL query level. Unauthorized chunks never enter the candidate set or reach the LLM.
- **Advanced Hybrid Search with Reciprocal Rank Fusion (RRF)**:
  - **Sparse Lexical Stream**: Tokenized BM25-style scoring matching headings, section titles, and body keywords.
  - **Dense Semantic Stream (`app/embeddings.py`)**: Vector embeddings generated via Google Gemini `text-embedding-004` (with LRU memory caching) paired with a subword CPU vectorizer fallback.
  - **Reciprocal Rank Fusion**: Merges sparse and dense ranked candidate lists using $RRF(d) = \sum \frac{1}{60 + rank(d)}$.
  - **Stage-2 Contextual Precedence Reranker**: Adjusts candidate scores dynamically based on Authority (policies outrank FAQs), Scope (regional addenda outrank global), and Recency.
- **3,000-Token Corpus Gate (Spec 4.3)**: `retrieve_chunks()` packs selected chunks in descending fusion order strictly up to a 3,000-token ceiling, calculated using `tiktoken` `cl100k_base`. Every LLM call records the exact number of corpus tokens in structured JSON logs.
- **Security & Attribution Guardrails (`app/guardrails.py`)**: Multi-layer input inspection detecting prompt injections and bulk-data exfiltration, combined with an automated post-generation citation attribution validator that strips any hallucinated citations not grounded in retrieved chunks.
- **Agentic Orchestration via LangGraph (`app/agent_graph.py`)**: A 7-node cyclical state machine graph (`guardrail` -> `confirmation` -> `router` -> `tool` / `retriever` -> `generator` -> `crag_verifier`). Incorporates Corrective RAG (CRAG) reflection: if generation produces ungrounded claims or hallucinated locators, the graph either self-corrects or deterministically returns the `no_answer` route.

## 3. Implementation of Section 5 Rules (Eligibility & Precedence)
All eligibility and precedence logic is implemented as pure, deterministic Python code in `app/policy.py`, independent of LLM whims:
1. **Eligibility (`is_eligible`)**: Requires `can_read(role, doc.access_level)` AND `region_matches(doc, region)` AND `doc.effective_date <= ref_date`.
2. **Precedence Cascade (`resolve`)**:
   - **Authority**: `policy`, `addendum`, `memo`, `handbook`, `calendar` outrank `faq` and `wiki`. Outdated FAQs are discarded in favor of authoritative policies.
   - **Scope**: Specific regional addenda (`IN`, `US`, `DE`) outrank `GLOBAL` documents for employees in those regions.
   - **Recency**: A later `effective_date` outranks an earlier one.
3. **Unresolved Conflicts**: If documents tie on all tests and assert differing values, `resolve()` sets `conflict=True`. The system quotes both values, cites both documents, and states that they conflict without arbitrarily picking one.
4. **Upcoming Changes (`upcoming_changes`)**: Evaluates documents whose `effective_date > ref_date`. If a future document is readable by the user, applies to their region, and would win on its effective date while changing the answer, it is reported as an upcoming change with its effective date and citation.

## 4. Access Control Enforcement Map
Access control is enforced strictly in application code, never relying on prompt instructions or LLM parameters:
| File | Function | Responsibility |
|---|---|---|
| `app/api.py` | `chat()` | Header check: missing or unverified `X-Employee-Id` returns HTTP 401 via `db.get_identity`. |
| `app/api.py` | `chat()` | Session hijacking prevention: returns HTTP 403 via `db.claim_session` if session belongs to another employee. |
| `app/retrieval.py` | `get_candidate_docs()` | Document isolation: excludes documents where `can_read(role, access_level)` is False before retrieval. |
| `app/tools.py` | `_resolve_target()` | Leave data view permission: verifies whether target is self, manager's direct report, or HR single-employee lookup. |
| `app/tools.py` | `get_team_leave()` | Manager role verification: non-managers receive immediate forbidden error. |
| `app/tools.py` | `apply_leave()`, `cancel_leave()` | Write authorization: restricted strictly to requester's own employee ID; re-verified inside write transactions. |

## 5. Personal Data Protection in Logs and Prompts
- **Identity Isolation (`app/db.py`)**: The `Identity` dataclass stores only non-PII attributes (`employee_id`, `role`, `region`, `level`, `manager_id`, `join_date`). Full names, emails, and phone numbers are excluded from the runtime identity object.
- **Log Scrubber (`app/logging_setup.py`)**: A custom `ScrubbingFilter` processes all console and file log streams. It scrubs phone numbers (`\+?\d[\d -]{8,}\d`), emails, and registered employee names loaded at startup via `db.load_sensitive_literals()`. Employee IDs and dates are preserved for auditability.
- **LLM Prompt Minimization (`app/gateway.py`)**: Prompts only receive the requester's role, region, and relevant policy text. No employee phone numbers or unauthorized personal records are ever forwarded to the LLM. Outgoing answers mask email addresses as `r***@domain`.

## 6. Failure Modes Found and Fixed
1. **Side-by-Side Tables Quoting Superseded Values**: In `POL-LEAVE-GLOBAL-V2` Section 3, a comparison table lists Version 1 (10 days) beside Version 2 (12 days). An unconstrained LLM quoted the Version 1 number. **Fix**: Dedicated system prompt constraints, rule tagger mapping, and targeted integration tests (`test_side_by_side_tables.py`) enforce quoting the active Version 2 value.
2. **Outdated FAQs Overriding Newer Policies**: `FAQ-LEAVE` stated 10 sick days while v2 policy stated 12. **Fix**: The Section 5 precedence engine enforces that policy outranks FAQ on authority before recency is evaluated.
3. **Ambiguous and Past Dates in Write Actions**: Users providing phrases like "next Friday" or dates in the past. **Fix**: `app/dates.py` implements a natural language date resolver. Past dates or ambiguous expressions trigger the `clarify` route rather than generating invalid proposals.
4. **Concurrent Double-Cancellation and Balance Tampering**: A repeated "confirm" command could attempt double refunding. **Fix**: `cancel_leave` and `apply_leave` execute inside `BEGIN IMMEDIATE` database transactions with status predicates (`status IN ('pending', 'approved')`), guaranteeing idempotency and race condition safety.
5. **Session Hijacking**: An employee attempting to send messages in an active session owned by another employee. **Fix**: `claim_session()` executes atomic insertion on primary key in `chat_sessions`, returning HTTP 403 on conflict.

## 7. Scaling to 10,000 Documents: Bottlenecks and Solutions
If the corpus grew to 10,000 documents (~100,000 chunks):
1. **Startup Ingestion Bottleneck**: Parsing 10,000 files sequentially on container startup would exceed the 10-minute timeout.
   - **Fix**: Decouple ingestion into an asynchronous offline pipeline or background worker. Store document content hashes (`content_hash`) in SQLite/PostgreSQL to perform incremental ingestion—only re-parsing added or modified files.
2. **Linear Chunk Scan in Retrieval**: Scanning and scoring all eligible chunks in SQLite table scans would exceed the 25-second response deadline.
   - **Fix**: Deploy a dedicated vector database (e.g. Qdrant, pgvector, or Chroma) with an HNSW index, paired with SQLite FTS5 for hybrid BM25 + dense embedding retrieval.
3. **Pre-Filtering at Scale**: Filtering 100,000 chunks in Python after fetching candidate metadata is inefficient.
   - **Fix**: Push access control metadata (`access_level`, `regions`, `effective_date`) directly into vector payload indexes, enabling single-pass filtered vector search (`role IN [...] AND region IN [...]`).
4. **Context Window Saturation**: With many matching documents, top-k chunks may saturate the 3,000-token gate without covering all relevant nuances.
   - **Fix**: Implement a two-stage retrieval pipeline: stage 1 retrieves top 50 chunks; stage 2 applies a cross-encoder reranker to select the top 5 most informative chunks within the 3,000-token budget.

# Insurance Claims AI Assistant

**Version 2.0 — General PDF Ingestion · Local MCP + LangGraph + RAG Prototype**

## Overview

Insurance Claims AI Assistant is a locally running Python prototype that helps insurance underwriters review medical evidence and validate specialist referrals associated with insurance claims.

The system combines deterministic validation rules, retrieval-augmented generation (RAG), Model Context Protocol (MCP), and a LangGraph agent powered by Ollama.

Its primary use case is determining whether available medical records support a specialist referral that occurred before a particular treatment.

**This is a decision-support prototype. It does not approve or deny insurance claims.**

## Architecture

```text
User
  |
  v
LangGraph Agent (langgraph_rag_agent.py)
  |
  +-- Ollama Local LLM
  |     +-- Tool selection
  |     +-- Natural-language responses
  |
  +-- MCP Client
        |
        v
      MCP Server (rag_server.py)
        +-- SQLite claim information and completed PDF registrations
        +-- Local PDF ingestion CLI (pdf_ingestion.py)
        |     +-- PyMuPDF page extraction / Tesseract OCR
        |     +-- Content hashing and incremental Ollama embeddings
        |     +-- SQLite ingestion completion gate
        +-- ChromaDB vector retrieval
        |     +-- Indexed medical PDF documents
        +-- Deterministic referral validator
              +-- Referral presence
              +-- Referral timing
              +-- Anatomy matching
              +-- Specialist verification

Additional components:
  - Structured JSON audit logging
  - Environment-based configuration
  - Operational health checks
  - Automated regression tests
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core application |
| Ollama | Local language-model inference |
| LangGraph | Agent orchestration |
| LangChain Core | Agent message representations |
| MCP SDK | Model Context Protocol communication |
| ChromaDB | Medical-document semantic retrieval |
| SQLite | Structured insurance claim data |
| PyMuPDF | PDF document processing |
| Tesseract 5 (Homebrew) | Local English OCR for scanned/image pages |
| JSON Schema | Tool argument validation |
| HTTPX | HTTP communication and timeout handling |
| unittest | Automated testing |

### Verified Python Package Versions

```text
ollama==0.6.3
langgraph==1.2.16
langchain-core==1.6.12
mcp==2.3.0
chromadb==1.5.9
PyMuPDF==1.28.2
jsonschema==4.26.0
httpx==0.28.1
```

These versions reflect the tested development environment, not a compatibility guarantee for every operating system or Python release. Pandas is not required by the verified prototype.

## Local Ollama Models

The tested environment contains:

- `ALIENTELLIGENCE/insuranceservices:latest` — default language model
- `llama3.2:latest` — alternative language model
- `embeddinggemma:latest` — embedding model for document retrieval

The tested configuration uses local inference without a hosted language-model API.

## Main Application Files

| File | Purpose |
|---|---|
| `langgraph_rag_agent.py` | Primary interactive AI agent |
| `rag_server.py` | MCP server exposing insurance and medical-document tools |
| `referral_validator.py` | Deterministic specialist-referral validation |
| `build_vector_index.py` | Incrementally refreshes the legacy medical-document index |
| `pdf_ingestion.py` | General PDF ingestion API and CLI |
| `demo_pdf_ingestion.py` | Isolated integration demonstration with real local services |
| `test_pdf_ingestion.py` | Synthetic PDF, OCR, failure recovery and claim-isolation tests |
| `generate_medical_documents.py` | Generates synthetic medical documents |
| `init_database.py` | Initializes the prototype claim database |
| `health_checks.py` | Operational health checks and CLI |
| `audit_logger.py` | Structured JSON audit logging |

The directory also includes earlier experimental agent and MCP implementations.

## MCP Tools

The primary MCP server exposes six tools:

1. `get_claim_status`
2. `get_claim_amount`
3. `list_open_claims`
4. `get_claim_medical_evidence`
5. `search_medical_documents`
6. `validate_claim_referral`

The LangGraph agent selects tools according to the user's question. For specialist-referral questions, it calls `validate_claim_referral` and treats the deterministic validation result as authoritative for this prototype.

## Running the Application

### Prerequisites

- Python 3.12
- An activated virtual environment with the packages listed above
- Ollama installed and running locally
- The configured Ollama model installed
- Initialized SQLite claim data and a built ChromaDB vector index

Project directory (relative to repository root):

```text
AI-Development/mcp-servers/insurance-claims/
```

From that directory, inspect system readiness:

```bash
python health_checks.py
```

Expected healthy report:

```json
{
  "status": "HEALTHY",
  "services": {
    "ollama": {
      "service": "ollama",
      "status": "HEALTHY",
      "model": "ALIENTELLIGENCE/insuranceservices:latest"
    },
    "mcp": {
      "service": "mcp",
      "status": "HEALTHY",
      "tool_count": 6,
      "missing_tools": []
    }
  }
}
```

Health-check exit codes: `0` for `HEALTHY`; `1` for `DEGRADED`.

Start the interactive agent:

```bash
python langgraph_rag_agent.py
```

Example interaction:

```text
You: Validate the specialist referral for CLM001
```

Enter `exit` or `quit` to stop the agent.

### Data preparation

The repository includes `init_database.py`, `generate_medical_documents.py`, and `build_vector_index.py` for preparing the synthetic prototype's data. Their exact arguments and required ordering should be checked in those scripts before preparing a fresh installation; the already-tested development environment has its data and index initialized.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `INSURANCE_OLLAMA_MODEL` | `ALIENTELLIGENCE/insuranceservices:latest` | Ollama model |
| `INSURANCE_RECURSION_LIMIT` | `13` | LangGraph recursion limit |
| `INSURANCE_OLLAMA_TIMEOUT` | `120.0` | Ollama request timeout (seconds) |

Example override:

```bash
INSURANCE_OLLAMA_MODEL=llama3.2 \
INSURANCE_RECURSION_LIMIT=20 \
INSURANCE_OLLAMA_TIMEOUT=180 \
python langgraph_rag_agent.py
```

Configuration validation rejects empty model names, invalid recursion limits, and nonpositive or nonfinite timeout values.

## Referral Validation

The deterministic validator examines available evidence for:

- Presence of specialist-referral documentation
- Whether the referral predates treatment
- Matching body part and anatomical laterality
- Documentation of an appropriate specialist

Findings include `PASS` and `REVIEW`. `REVIEW` signals the need for human examination or clarification, not automatic denial of insurance coverage.

### Verified Demonstration Scenarios

| Claim | Finding | Explanation |
|---|---|---|
| CLM001 | PASS | Preceding referral, matching RIGHT knee anatomy, orthopaedic specialist documented |
| CLM002 | REVIEW | Treatment anatomy is insufficiently documented |
| CLM003 | REVIEW | No referral found in available indexed evidence |
| CLM004 | REVIEW | LEFT ankle referral versus RIGHT ankle treatment |

Additional tests cover unknown claim IDs, invalid claim identifiers, and missing claim IDs.

### Example: CLM001

```text
Overall finding: PASS
Referral presence: PASS
Referral date: PASS
Anatomy: PASS
Specialist: PASS
```

Referral dated **January 5, 2026**, before treatment on **January 9, 2026**; both records concern the **RIGHT knee**. Named specialist: **Dr. A. Malik (Orthopaedic Specialist)**. Supporting documents: `CLM001-DOC02`, `CLM001-DOC03`.

### Example: CLM004

```text
Overall finding: REVIEW
Referral presence: PASS
Referral date: PASS
Anatomy: REVIEW
Specialist: PASS
```

The referral concerns the **LEFT ankle**, while treatment concerns the **RIGHT ankle**. Manual clarification is required.


## Version 2.0: Ingest a Previously Unseen PDF

Activate the existing Python 3.12 environment from this project directory:

```bash
cd ~/AI-Development/mcp-servers/insurance-claims
source ../../.venv/bin/activate
```

The only new runtime dependency is the local Tesseract executable. No Python package
was added. On Apple Silicon macOS:

```bash
brew install tesseract
tesseract --version
ollama list
```

`embeddinggemma` must already be installed in the running local Ollama service.
The Homebrew Tesseract package includes English language data. OCR is performed
locally through Tesseract's stdin/stdout interface; no hosted OCR or inference is used.
See the [Tesseract command-line documentation](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).

Ingest any local PDF path, explicitly associating it with an **existing** claim:

```bash
python pdf_ingestion.py "/absolute/path/new-medical-document.pdf" --claim-id CLM001
```

No predefined filename or PDF template is required for ingestion or search. Claim IDs
are normalized to uppercase and must follow `CLM` plus three digits, matching an
existing SQLite `claims` row. Ingestion never creates a claim or trusts an ID inside
the PDF to select its owner. One identical PDF may be explicitly attached to two
claims; each attachment has separate claim-scoped IDs and vectors.

A successful JSON response reports `complete`, the document ID, page/chunk counts,
and OCR page numbers. Repeating identical bytes for the same claim, including after
renaming the file, returns `duplicate` without adding vectors or registrations.
Changed PDF bytes create a distinct document; both versions remain available and
multiple referral/treatment candidates require manual pairing.

### Extraction and indexing

- PyMuPDF extracts selectable text page by page. Pages with insufficient or broken
  text, and pages containing raster images alongside selectable text, are rendered
  at 300 DPI and OCRed in full with English Tesseract. This covers mixed PDFs and
  scanned evidence beneath selectable headers.
- Every page must yield usable text. Blank, unreadable, corrupted and locked PDFs
  fail explicitly, with the original page number where applicable. A blank page is
  conservatively treated as incomplete extraction, rather than silently skipped.
- Text uses the existing 800-character chunks with 150-character overlap.
  `embeddinggemma` supplies embeddings to the existing
  `insurance_medical_documents` Chroma collection via stable-ID upserts.
- Metadata includes claim/document IDs, SHA-256, source filename and absolute path,
  original one-based PDF page, extraction method, title, and detected type/date.
- SQLite's existing `claim_documents` table is retained. A lazily created
  `pdf_ingestions` table records pending, failed and completed jobs, counts, source
  hashes and actionable extraction errors. Existing registrations remain intact.
- All vectors are written before SQLite publishes the registration and completion
  together. Interrupted/failed vectors remain unpublished and are excluded by
  backend retrieval. Retrying the same PDF safely upserts those same IDs. Writers
  are serialized by SQLite; this is a synchronous local prototype.
- Failed/pending jobs are disclosed as `incomplete_ingestions`, and prevent a
  confident validation `PASS` for that claim. Retry the same PDF after fixing the
  missing OCR/model/service. No automatic cleanup deletes prior data.

Missing files or unreadable filesystem paths are rejected before a content hash/job
can be created. Such requests return an error immediately and must be resubmitted;
they do not register an empty document. SQLite/index outages also surface as errors.
Original PDFs are read only; the index retains extracted text and source hashes,
but does not make archival PDF copies. Keep source files for later human review.

### Retrieval and deterministic findings

All six MCP tool names/signatures remain available. Ingestion is a local CLI/API,
so existing LangGraph tool discovery and health checks need no orchestration changes.
A running MCP process reads completed registrations on each call and can see newly
ingested PDFs without restarting or rebuilding the collection.

Use the interactive agent:

```bash
python langgraph_rag_agent.py
```

Example questions:

```text
Find medical evidence about right knee treatment for CLM001.
Validate the specialist referral for CLM001.
```

For direct local search or deterministic validation:

```bash
python search_documents.py
python -c 'import json; from rag_server import validate_claim_referral; print(json.dumps(validate_claim_referral("CLM001"), indent=2))'
```

The standalone search now also requires a claim ID. Backend retrieval checks claim
existence and restricts results to that claim's registered, completed documents;
semantic queries apply these restrictions in Chroma itself. No unfiltered medical
search is exposed by these workflows.

Findings retain document IDs and add `citations` containing real source paths,
original page numbers, passage text, chunk IDs, claim IDs and extraction methods.
The deterministic agent formatter displays filename/page/text citations without
LLM rewriting. Citations identify the evidence examined; they are not independent
proof of clinical correctness.

General PDF ingestion does **not** imply general clinical-language understanding.
For dynamic pages, conservative rules recognize explicit `Document type: Referral`
or `Document type: Treatment` labels, or matching headings; dates require an
unambiguous valid ISO `Document date`, `Referral date`, or `Treatment date` label.
For example, this supported evidence is sufficient to exercise the existing rules:

```text
Document type: Referral
Document date: 2026-01-05
Issued by Dr. A. Malik, Orthopaedic Specialist
I refer this fictional patient for RIGHT knee physiotherapy.
```

A corresponding treatment page can contain:

```text
Document type: Treatment
Document date: 2026-01-09
Physiotherapy commenced for RIGHT knee rehabilitation.
```

Unknown layouts remain indexed/searchable and produce `REVIEW` when the rules cannot
establish the needed facts. No dates, authors or roles are inferred by an LLM.
Different page types/dates within one packet can participate separately. Missing or
conflicting dates, multiple candidates, ambiguous anatomy or specialist identities
require review. The existing specialist check still recognizes named orthopaedic
specialists in `Issued by Dr. ..., Orthopaedic Specialist` form; anatomy support
remains knee/shoulder/ankle with LEFT/RIGHT laterality. Existing Version 1 date-check
`FAIL` semantics for a late referral remain unchanged.

OCR passages always require manual transcription verification and prevent an overall
`PASS`, even when the remaining deterministic checks succeed. English printed text
is tested; handwriting, low-resolution scans, tables, unusual encodings and other
languages may need better source documents. Text usability checks are heuristics,
not a clinical completeness guarantee. All uncertainty remains a human review task.

### Isolated storage and Version 2.0 verification

Both MCP retrieval and ingestion honor optional absolute paths:

```bash
export INSURANCE_DB_PATH="/absolute/path/test/claims.db"
export INSURANCE_CHROMA_PATH="/absolute/path/test/chroma"
```

The selected SQLite database must already contain the claims table and target claim.
Leave these variables unset to use the existing `data/` paths. Never point an
isolated test at the demonstration database. The new tests create their own claims,
PDFs, database and Chroma directory under temporary storage.

Run the new automated suite (actual Tesseract and Chroma, fake embeddings; no Ollama
needed) and the real-service demonstration:

```bash
python -m unittest test_pdf_ingestion -v
python demo_pdf_ingestion.py
```

The demonstration uses actual local Ollama embeddings, Chroma, MCP stdio and the
existing LangGraph workflow. It verifies text/scanned ingestion, duplicate handling,
SQLite registration, cross-claim isolation, deterministic findings and genuine page
citations. It also compares the original claims, registrations, PDF hashes and
vector contents before/after. Temporary synthetic demonstration storage is removed
when it completes. Existing test commands below still run the full Version 1 suite.

## Audit Logging

Structured JSON Lines audit records are stored in:

```text
logs/audit.jsonl
```

Events include `VALIDATION_STARTED`, `VALIDATION_COMPLETED`, and `VALIDATION_FAILED`. Audit metadata includes request identifiers, claim IDs, model names, validation results, duration, and standardized error codes. Raw medical-document contents and exception messages are not intentionally included in these records.

HTTP client timeouts are classified as `OLLAMA_TIMEOUT`.

The audit trail supports debugging and execution tracing; it is not a formally validated regulatory audit system.

## Health Checks

Checks cover Ollama connectivity, configured-model availability, MCP server initialization, required MCP tool discovery, and combined system readiness. The primary interactive agent also checks startup readiness and blocks startup when the configured model or required tools are missing.

A `HEALTHY` result verifies connectivity and tool availability, **not** clinical accuracy, database integrity, or guaranteed successful inference.

## Formatting and linting

Black and Flake8 are configured for all Python files in this project, including
examples and tests. Run these commands from the project directory:

```bash
cd ~/AI-Development/mcp-servers/insurance-claims
source ../../.venv/bin/activate
python -m pip install -r requirements-dev.txt  # One-time development setup
make format  # Apply Black formatting
make lint    # Check Black formatting and run Flake8
```

`make format` applies Black; `make lint` checks formatting without modifying files
and runs Flake8. The Makefile defaults to the existing `../../.venv/bin/python`;
use `make lint PYTHON=/path/to/python` to select another environment.
Equivalent direct commands are:

```bash
python -m black .
python -m black --check .
python -m flake8 .
```

Black settings are in `pyproject.toml` (Python 3.12, 88-character target).
Flake8 settings are in `.flake8`. Data, logs, virtual environments and bytecode
caches are excluded. Black controls line wrapping; Flake8's E501 and E203 are
ignored to accommodate long strings/comments and slice formatting, following
[Black's interoperability guidance](https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html).
Other default Flake8 checks remain enabled, including unused imports and undefined
names. Pinned development tool versions are in `requirements-dev.txt`.

## Automated Tests

The Version 2.0 verification passed all **42 existing Version 1 tests** and
**26 new PDF ingestion tests** (68 total). The separate real-service demonstration
also passed: actual Ollama embeddings, MCP retrieval, LangGraph validation, OCR,
claim isolation and citations. Original demonstration claims, 15 document
registrations/PDFs and 15 vectors were compared with their pre-change snapshot and
remained unchanged. New tests additionally verify that incremental ingestion and
the legacy index builder preserve each other's indexed documents.

| Test module | Tests |
|---|---:|
| `test_referral_regression.py` | 5 |
| `test_referral_formatter.py` | 4 |
| `test_audit_logger.py` | 4 |
| `test_agent_reliability.py` | 4 |
| `test_agent_config.py` | 3 |
| `test_health_checks.py` | 15 |
| `test_agent_e2e.py` | 7 |
| **Total** | **42** |

Run the verified test suite:

```bash
python test_referral_regression.py
python test_referral_formatter.py
python test_audit_logger.py -v
python test_agent_reliability.py -v
python test_agent_config.py -v
python test_health_checks.py -v
python test_agent_e2e.py
```

End-to-end and certain health-check tests require the appropriate local services.

## Privacy and Limitations

**This is a research and demonstration prototype using synthetic medical and insurance data.**

- It is not a production insurance claims-processing system.
- It does not issue final coverage, eligibility, approval, or denial decisions.
- Missing indexed evidence does not prove a document never existed elsewhere.
- Clinical and claims-validation outcomes have not been independently validated on real cases.
- Insurance rules vary by policy, jurisdiction, and claim type.
- Local execution alone does not ensure complete security, access control, or regulatory compliance.
- Real patient information would require further privacy, security, governance, and compliance safeguards.
- The legacy index builder now upserts registered legacy documents without deleting the collection. Dynamic PDFs must use the ingestion CLI.

## Version 1.0 Completion Criteria

The agreed prototype demonstration goals have been verified: local LLM-based agent execution, MCP integration, SQLite claim-data access, ChromaDB medical-document retrieval, deterministic specialist-referral validation, evidence-grounded reporting, JSON audit logging, configurable execution, health checks, automated regressions, and interactive demonstrations of CLM001–CLM004.

**Version 1.0 functionality is retained in Version 2.0.**

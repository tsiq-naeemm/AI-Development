# Insurance Claims AI Assistant

**Version 1.0 — Local MCP + LangGraph + RAG Prototype**

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
        +-- SQLite claim information
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
| `build_vector_index.py` | Builds the medical-document vector index |
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

## Automated Tests

The final version 1.0 checkpoint passed **42 automated tests**.

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
- Regenerating synthetic data and the vector index relies on the project's existing preparation scripts.

## Version 1.0 Completion Criteria

The agreed prototype demonstration goals have been verified: local LLM-based agent execution, MCP integration, SQLite claim-data access, ChromaDB medical-document retrieval, deterministic specialist-referral validation, evidence-grounded reporting, JSON audit logging, configurable execution, health checks, automated regressions, and interactive demonstrations of CLM001–CLM004.

**Status: Version 1.0 prototype functionality verified.**

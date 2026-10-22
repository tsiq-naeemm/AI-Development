"""Phase 5D: SQLite claims + local ChromaDB evidence exposed via MCP.
Run as a stdio MCP server, not directly as a terminal prompt.
"""
import json
import re
import sqlite3
from pathlib import Path

import chromadb
from mcp.server import MCPServer
from ollama import Client
from referral_validator import (
    validate_referral_presence, validate_referral_date, validate_anatomy,
    validate_referring_specialist, combine_validation_results,
)

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'data' / 'insurance_claims.db'
CHROMA_PATH = ROOT / 'data' / 'chroma_db'
COLLECTION_NAME = 'insurance_medical_documents'
EMBED_MODEL = 'embeddinggemma'

server = MCPServer('InsuranceClaimsRAG')


def connect_db():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f'Initialize SQLite first: {DB_PATH}')
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@server.tool()
def get_claim_status(claim_id: str) -> str:
    """Get the administrative status of a claim, e.g. CLM001."""
    with connect_db() as conn:
        row = conn.execute('SELECT status FROM claims WHERE claim_id = ?', (claim_id,)).fetchone()
    return row['status'] if row else 'Claim not found'


@server.tool()
def get_claim_amount(claim_id: str) -> dict:
    """Get the amount and currency for one claim."""
    with connect_db() as conn:
        row = conn.execute('SELECT claim_id, amount, currency FROM claims WHERE claim_id = ?', (claim_id,)).fetchone()
    return dict(row) if row else {'error': 'Claim not found'}


@server.tool()
def list_open_claims() -> list[dict]:
    """List claims whose administrative status is Open."""
    with connect_db() as conn:
        rows = conn.execute("SELECT claim_id, status, amount FROM claims WHERE status = 'Open' ORDER BY claim_id").fetchall()
    return [dict(row) for row in rows]


def get_collection():
    """Open the existing ChromaDB collection."""
    if not CHROMA_PATH.is_dir():
        raise FileNotFoundError(f"ChromaDB index not found: {CHROMA_PATH}")
    return chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
        name=COLLECTION_NAME
    )


@server.tool()
def get_claim_medical_evidence(claim_id: str) -> dict:
    """
    Retrieve all indexed medical evidence for a claim.

    Unlike semantic search, this returns every matching
    passage without similarity ranking.
    """
    claim_id = claim_id.strip().upper()

    if not (
        len(claim_id) == 6
        and claim_id.startswith("CLM")
        and claim_id[3:].isdigit()
    ):
        raise ValueError(
            "Invalid claim ID. Expected format: CLM001"
        )

    collection = get_collection()

    records = collection.get(
        where={"claim_id": claim_id},
        include=["documents", "metadatas"],
    )

    results = []

    for chunk_id, document, metadata in zip(
        records["ids"],
        records["documents"],
        records["metadatas"],
    ):
        results.append({
            "chunk_id": chunk_id,
            "document_id": metadata["document_id"],
            "document_type": metadata["document_type"],
            "document_date": metadata["document_date"],
            "page": metadata["page"],
            "source": metadata["file_path"],
            "text": document,
        })

    results.sort(
        key=lambda item: (
            item["document_date"],
            item["document_id"],
            item["page"],
            item["chunk_id"],
        )
    )

    return {
        "claim_id": claim_id,
        "passage_count": len(results),
        "results": results,
    }

@server.tool()
def search_medical_documents(claim_id: str, question: str, top_k: int = 5) -> dict:
    """Retrieve source-cited medical evidence for one claim. For referral validation,
    include referral, treatment and consultation documents. Never infer that missing
    evidence proves a referral did not exist outside the supplied records.
    """
    claim_id = claim_id.strip().upper()
    if not question.strip():
        return {'error': 'Question cannot be empty'}
    if not 1 <= top_k <= 15:
        return {'error': 'top_k must be between 1 and 15'}
    with connect_db() as conn:
        claim = conn.execute('SELECT claim_id FROM claims WHERE claim_id = ?', (claim_id,)).fetchone()
        if not claim:
            return {'error': f'Claim {claim_id} not found'}
        registered = [dict(r) for r in conn.execute(
            'SELECT document_id, document_type, document_date, file_path FROM claim_documents WHERE claim_id = ? ORDER BY document_id',
            (claim_id,)).fetchall()]
    if not registered:
        return {'claim_id': claim_id, 'registered_document_count': 0, 'results': [], 'warning': 'No registered documents for this claim'}
    if not CHROMA_PATH.exists():
        return {'error': 'ChromaDB index missing. Run build_vector_index.py'}
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
    indexed_count = collection.get(where={'claim_id': claim_id}, include=[])['ids']
    if not indexed_count:
        return {'error': f'No indexed passages for {claim_id}. Rebuild vector index.'}
    embedding = Client().embed(model=EMBED_MODEL, input=question).embeddings[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, len(indexed_count)),
        where={'claim_id': claim_id},
        include=['documents', 'metadatas', 'distances'],
    )
    passages = []
    for chunk_id, content, meta, distance in zip(
        results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]
    ):
        passages.append({
            'chunk_id': chunk_id,
            'document_id': meta['document_id'],
            'document_type': meta['document_type'],
            'document_date': meta['document_date'],
            'page': meta['page'],
            'source': meta['file_path'],
            'text': content,
            'cosine_distance': round(float(distance), 4),
        })
    return {
        'claim_id': claim_id,
        'registered_document_count': len(registered),
        'indexed_passage_count': len(indexed_count),
        'results': passages,
        'note': 'Semantic similarity ranks passages; it does not establish referral validity. Cite document_id and page. For validation, compare referral and treatment dates, anatomy and specialist, and disclose missing/conflicting evidence.',
    }


@server.tool()
def validate_claim_referral(claim_id: str) -> dict:
    """Deterministically check documented referral presence, timing, anatomy and specialist.

    Findings concern the indexed evidence only; they are not coverage decisions.
    """
    claim_id = claim_id.strip().upper()
    with connect_db() as conn:
        claim = conn.execute(
            "SELECT 1 FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
    if claim is None:
        return {
            "claim_id": claim_id,
            "error": "CLAIM_NOT_FOUND",
            "explanation": f"Claim {claim_id} does not exist in the claims database.",
        }

    evidence = get_claim_medical_evidence(claim_id)
    claim_id = evidence["claim_id"]
    passages = evidence["results"]
    if not passages:
        return {"claim_id": claim_id, "overall_finding": "REVIEW",
                "explanation": "No indexed medical evidence available.", "checks": {}}

    grouped = {}
    for passage in passages:
        key = passage["document_id"]
        if key not in grouped:
            grouped[key] = {"type": passage["document_type"].strip().lower(),
                            "date": passage["document_date"], "texts": [], "pages": []}
        grouped[key]["texts"].append(passage["text"])
        grouped[key]["pages"].append(passage["page"])

    referrals = [(doc_id, doc) for doc_id, doc in grouped.items() if doc["type"] == "referral"]
    treatments = [(doc_id, doc) for doc_id, doc in grouped.items() if doc["type"] == "treatment"]
    checks = {"referral_presence": validate_referral_presence([d["type"] for d in grouped.values()])}

    def review(message):
        return {"finding": "REVIEW", "explanation": message}

    if not referrals:
        checks["referral_date"] = review("No referral available for date comparison.")
        checks["anatomy"] = review("No referral available for anatomy comparison.")
        checks["specialist"] = review("No referral available for specialist verification.")
    elif not treatments:
        checks["referral_date"] = review("No treatment available for date comparison.")
        checks["anatomy"] = review("No treatment available for anatomy comparison.")
    elif len(referrals) != 1 or len(treatments) != 1:
        checks["referral_date"] = review("Multiple referral or treatment documents require manual pairing.")
        checks["anatomy"] = review("Multiple referral or treatment documents require manual pairing.")
    else:
        referral_id, referral = referrals[0]
        treatment_id, treatment = treatments[0]
        checks["referral_date"] = validate_referral_date(referral["date"], treatment["date"])
        anatomy_pattern = re.compile(r"\b(LEFT|RIGHT)\s+(knee|shoulder|ankle)\b", re.I)
       
        def anatomy(doc, document_type):
            text = " ".join(doc["texts"])

            if document_type == "treatment":
                match = re.search(
                    r"Physiotherapy commenced for\s+"
                    r"(LEFT|RIGHT)\s+(knee|shoulder|ankle)",
                    text,
                    re.IGNORECASE,
                )
                if match:
                    return f"{match.group(1).upper()} {match.group(2).lower()}"

            matches = {
                f"{m.group(1).upper()} {m.group(2).lower()}"
                for m in anatomy_pattern.finditer(text)
            }
            return next(iter(matches)) if len(matches) == 1 else ""
                
        checks["anatomy"] = validate_anatomy(anatomy(referral, "referral"),anatomy(treatment, "treatment"),)
        checks["referral_date"]["source_document_ids"] = [referral_id, treatment_id]
        checks["anatomy"]["source_document_ids"] = [referral_id, treatment_id]

    if referrals:
        if len(referrals) != 1:
            checks["specialist"] = review("Multiple referrals require manual specialist review.")
        else:
            referral_id, referral = referrals[0]
            referral_text = " ".join(referral["texts"])
            match = re.search(r"Issued by\s+(Dr\.?\s+[A-Za-z.\s]+?),\s*(Orthopaedic Specialist|Orthopedic Specialist)", referral_text, re.I)
            checks["specialist"] = validate_referring_specialist(match.group(1).strip() if match else "", match.group(2).strip() if match else "")
            checks["specialist"]["source_document_ids"] = [referral_id]

    combined = combine_validation_results(checks)
    return {"claim_id": claim_id, "overall_finding": combined["overall_finding"],
            "checks": combined["checks"], "source_document_ids": sorted(grouped),
            "note": "Evidence validation only; not an insurance approval or denial. Missing evidence may exist elsewhere."}


if __name__ == '__main__':
    server.run()

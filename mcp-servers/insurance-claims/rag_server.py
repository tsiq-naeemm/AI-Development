"""Phase 5D: SQLite claims + local ChromaDB evidence exposed via MCP.
Run as a stdio MCP server, not directly as a terminal prompt.
"""

import os
import re
import sqlite3
from pathlib import Path

import chromadb
from mcp.server import MCPServer
from ollama import Client
from pdf_ingestion import require_claim, normalize_claim, ingestion_state
from referral_validator import (
    validate_referral_presence,
    validate_referral_date,
    validate_anatomy,
    validate_referring_specialist,
    combine_validation_results,
)

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("INSURANCE_DB_PATH", str(ROOT / "data/insurance_claims.db")))
CHROMA_PATH = Path(os.getenv("INSURANCE_CHROMA_PATH", str(ROOT / "data/chroma_db")))
COLLECTION_NAME = "insurance_medical_documents"
EMBED_MODEL = "embeddinggemma"


server = MCPServer("InsuranceClaimsRAG")


def connect_db():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Initialize SQLite first: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@server.tool()
def get_claim_status(claim_id: str) -> str:
    """Get the administrative status of a claim, e.g. CLM001."""
    with connect_db() as conn:
        row = conn.execute(
            "SELECT status FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
    return row["status"] if row else "Claim not found"


@server.tool()
def get_claim_amount(claim_id: str) -> dict:
    """Get the amount and currency for one claim."""
    with connect_db() as conn:
        row = conn.execute(
            "SELECT claim_id, amount, currency FROM claims WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
    return dict(row) if row else {"error": "Claim not found"}


@server.tool()
def list_open_claims() -> list[dict]:
    """List claims whose administrative status is Open."""
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT claim_id, status, amount FROM claims WHERE status = 'Open' ORDER BY claim_id"
        ).fetchall()
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
    with connect_db() as conn:
        claim_id = require_claim(conn, claim_id)
        allowed, incomplete = ingestion_state(conn, claim_id)
    if not allowed:
        return {
            "claim_id": claim_id,
            "passage_count": 0,
            "results": [],
            "incomplete_ingestions": incomplete,
        }

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
        if (
            metadata.get("claim_id") != claim_id
            or metadata.get("document_id") not in allowed
        ):
            continue
        results.append(
            {
                "claim_id": claim_id,
                "extraction_method": metadata.get("extraction_method", "text"),
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "document_type": metadata["document_type"],
                "document_date": metadata["document_date"],
                "page": metadata["page"],
                "source": metadata["file_path"],
                "text": document,
            }
        )

    results.sort(
        key=lambda item: (
            item["document_date"],
            item["document_id"],
            item["page"],
            int(item["chunk_id"].rsplit("-C", 1)[-1]),
        )
    )

    return {
        "claim_id": claim_id,
        "incomplete_ingestions": incomplete,
        "passage_count": len(results),
        "results": results,
    }


@server.tool()
def search_medical_documents(claim_id: str, question: str, top_k: int = 5) -> dict:
    """Retrieve source-cited medical evidence for one claim. For referral validation,
    include referral, treatment and consultation documents. Never infer that missing
    evidence proves a referral did not exist outside the supplied records.
    """
    if not question.strip():
        return {"error": "Question cannot be empty"}
    if not 1 <= top_k <= 15:
        return {"error": "top_k must be between 1 and 15"}
    claim_id = normalize_claim(claim_id)
    with connect_db() as conn:
        if not conn.execute(
            "SELECT 1 FROM claims WHERE claim_id=?", (claim_id,)
        ).fetchone():
            return {"error": f"Claim {claim_id} not found"}
        allowed, incomplete = ingestion_state(conn, claim_id)
    if not allowed:
        return {
            "claim_id": claim_id,
            "registered_document_count": 0,
            "results": [],
            "incomplete_ingestions": incomplete,
            "warning": "No completely ingested documents for this claim",
        }
    collection = get_collection()
    filters = {
        "$and": [{"claim_id": claim_id}, {"document_id": {"$in": sorted(allowed)}}]
    }
    indexed = collection.get(where=filters, include=[])["ids"]
    if not indexed:
        return {
            "claim_id": claim_id,
            "results": [],
            "error": "No indexed passages available",
            "incomplete_ingestions": incomplete,
        }
    embedding = (
        Client(timeout=120).embed(model=EMBED_MODEL, input=question).embeddings[0]
    )
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, len(indexed)),
        where=filters,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for chunk_id, content, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if meta.get("claim_id") != claim_id or meta.get("document_id") not in allowed:
            continue
        passages.append(
            {
                "claim_id": claim_id,
                "chunk_id": chunk_id,
                "document_id": meta["document_id"],
                "document_type": meta["document_type"],
                "document_date": meta["document_date"],
                "page": meta["page"],
                "source": meta["file_path"],
                "text": content,
                "extraction_method": meta.get("extraction_method", "text"),
                "cosine_distance": round(float(distance), 4),
            }
        )
    return {
        "claim_id": claim_id,
        "registered_document_count": len(allowed),
        "indexed_passage_count": len(indexed),
        "results": passages,
        "incomplete_ingestions": incomplete,
        "note": "Similarity does not establish referral validity. Cite document_id and page; missing evidence may exist elsewhere.",
    }


@server.tool()
def validate_claim_referral(claim_id: str) -> dict:
    """Deterministically check documented referral presence, timing, anatomy and specialist.

    Findings concern the indexed evidence only; they are not coverage decisions.
    """
    claim_id = normalize_claim(claim_id)
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

    grouped = {}
    for passage in passages:
        key = (
            passage["document_id"],
            passage["document_type"],
            passage["document_date"],
        )
        if key not in grouped:
            grouped[key] = {
                "document_id": passage["document_id"],
                "type": passage["document_type"].strip().lower(),
                "date": passage["document_date"],
                "texts": [],
                "pages": [],
                "dynamic": "-PDF-" in passage["document_id"],
            }
        grouped[key]["texts"].append(passage["text"])
        grouped[key]["pages"].append(passage["page"])

    referrals = [
        (doc["document_id"], doc)
        for doc in grouped.values()
        if doc["type"] == "referral"
    ]
    treatments = [
        (doc["document_id"], doc)
        for doc in grouped.values()
        if doc["type"] == "treatment"
    ]
    checks = {
        "referral_presence": validate_referral_presence(
            [d["type"] for d in grouped.values()]
        )
    }

    def review(message):
        return {"finding": "REVIEW", "explanation": message}

    if not referrals:
        checks["referral_date"] = review("No referral available for date comparison.")
        checks["anatomy"] = review("No referral available for anatomy comparison.")
        checks["specialist"] = review(
            "No referral available for specialist verification."
        )
    elif not treatments:
        checks["referral_date"] = review("No treatment available for date comparison.")
        checks["anatomy"] = review("No treatment available for anatomy comparison.")
    elif len(referrals) != 1 or len(treatments) != 1:
        checks["referral_date"] = review(
            "Multiple referral or treatment documents require manual pairing."
        )
        checks["anatomy"] = review(
            "Multiple referral or treatment documents require manual pairing."
        )
    else:
        referral_id, referral = referrals[0]
        treatment_id, treatment = treatments[0]
        try:
            checks["referral_date"] = validate_referral_date(
                referral["date"], treatment["date"]
            )
        except (ValueError, TypeError):
            checks["referral_date"] = review(
                "Document dates are missing, ambiguous or invalid."
            )
        anatomy_pattern = re.compile(r"\b(LEFT|RIGHT)\s+(knee|shoulder|ankle)\b", re.I)

        def anatomy(doc, document_type):
            text = " ".join(doc["texts"])

            if document_type == "treatment" and not doc["dynamic"]:
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

        checks["anatomy"] = validate_anatomy(
            anatomy(referral, "referral"),
            anatomy(treatment, "treatment"),
        )
        checks["referral_date"]["source_document_ids"] = [referral_id, treatment_id]
        checks["anatomy"]["source_document_ids"] = [referral_id, treatment_id]

    if referrals:
        if len(referrals) != 1:
            checks["specialist"] = review(
                "Multiple referrals require manual specialist review."
            )
        else:
            referral_id, referral = referrals[0]
            referral_text = " ".join(referral["texts"])
            matches = re.findall(
                r"Issued by\s+(Dr\.?\s+[A-Za-z.\s]+?),\s*(Orthopaedic Specialist|Orthopedic Specialist)",
                referral_text,
                re.I,
            )
            unique = set(matches)
            match = next(iter(unique)) if len(unique) == 1 else None
            checks["specialist"] = validate_referring_specialist(
                match[0].strip() if match else "", match[1].strip() if match else ""
            )
            checks["specialist"]["source_document_ids"] = [referral_id]

    checks["referral_presence"]["source_document_ids"] = sorted(
        {r[0] for r in referrals}
    )
    issues = evidence.get("incomplete_ingestions", [])
    uncertain = any(
        p.get("extraction_method") == "ocr" or p["document_type"] == "unknown"
        for p in passages
    )
    if issues or uncertain:
        checks["evidence_quality"] = review(
            "Incomplete ingestion, OCR transcription, or unclassified evidence requires manual verification."
        )
    combined = combine_validation_results(checks)
    if issues or uncertain:
        combined["overall_finding"] = "REVIEW"
    return {
        "claim_id": claim_id,
        "overall_finding": combined["overall_finding"],
        "checks": combined["checks"],
        "source_document_ids": sorted({p["document_id"] for p in passages}),
        "citations": passages,
        "incomplete_ingestions": issues,
        "note": "Evidence validation only; not an insurance approval or denial. Missing evidence may exist elsewhere.",
    }


if __name__ == "__main__":
    server.run()

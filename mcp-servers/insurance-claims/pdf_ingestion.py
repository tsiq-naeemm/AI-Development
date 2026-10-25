"""Local, incremental PDF ingestion. SQLite completion gates vector visibility."""

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess

import chromadb
import pymupdf as fitz
from ollama import Client
from build_vector_index import chunk_text, COLLECTION_NAME, EMBED_MODEL, BATCH_SIZE

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("INSURANCE_DB_PATH", str(ROOT / "data/insurance_claims.db")))
CHROMA_PATH = Path(os.getenv("INSURANCE_CHROMA_PATH", str(ROOT / "data/chroma_db")))


class IngestionError(ValueError):
    """An extraction/indexing failure, with no document published."""


def normalize_claim(claim_id):
    if not isinstance(claim_id, str) or not re.fullmatch(
        r"CLM\d{3}", claim_id.strip().upper()
    ):
        raise ValueError("Invalid claim ID. Expected format: CLM001")
    return claim_id.strip().upper()


def require_claim(conn, claim_id):
    claim_id = normalize_claim(claim_id)
    if not conn.execute(
        "SELECT 1 FROM claims WHERE claim_id=?", (claim_id,)
    ).fetchone():
        raise ValueError(f"Unknown claim ID: {claim_id}")
    return claim_id


def ensure_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS claim_documents (
        document_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL REFERENCES claims(claim_id),
        document_type TEXT NOT NULL, document_date TEXT NOT NULL,
        title TEXT NOT NULL, file_path TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS pdf_ingestions (
        document_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL REFERENCES claims(claim_id),
        sha256 TEXT NOT NULL, source TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('pending','complete','failed')),
        page_count INTEGER NOT NULL DEFAULT 0, chunk_count INTEGER NOT NULL DEFAULT 0,
        error TEXT NOT NULL DEFAULT '', UNIQUE(claim_id, sha256))""")


def usable(text):
    return len(re.findall(r"[A-Za-z0-9]", text)) >= 12 and "\ufffd" not in text


def ocr_page(page):
    executable = shutil.which("tesseract")
    if not executable:
        raise IngestionError(
            "Local OCR unavailable: install with brew install tesseract"
        )
    png = page.get_pixmap(dpi=300, alpha=False).tobytes("png")
    try:
        result = subprocess.run(
            [executable, "stdin", "stdout", "-l", "eng", "--psm", "3"],
            input=png,
            capture_output=True,
            timeout=120,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IngestionError(f"Local OCR failed ({type(exc).__name__})") from None
    return result.stdout.decode("utf-8").strip()


def page_fields(text):
    """Conservative labels/headings only; never guess dates from unrelated mentions."""
    kinds = set(
        re.findall(
            r"(?im)^\s*(?:Document\s+type\s*[:\n]?\s*)?(referral|treatment)\b", text
        )
    )
    kinds = {v.lower() for v in kinds}
    kind = next(iter(kinds)) if len(kinds) == 1 else "unknown"
    dates = set(
        re.findall(
            r"(?im)\b(?:Document|Referral|Treatment)\s+date\s*[:\n]?\s*(\d{4}-\d{2}-\d{2})\b",
            text,
        )
    )
    valid_date = ""
    if len(dates) == 1:
        candidate = next(iter(dates))
        try:
            valid_date = date.fromisoformat(candidate).isoformat()
        except ValueError:
            pass
    return kind, valid_date


def extract_pages(data):
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError(f"Cannot open PDF ({type(exc).__name__})") from None
    with pdf:
        if pdf.needs_pass:
            raise IngestionError("Encrypted PDF requires an unlocked local copy")
        if not pdf.is_pdf or not pdf.page_count:
            raise IngestionError("No PDF pages available")
        pages = []
        for index, page in enumerate(pdf, 1):
            try:
                text = page.get_text("text", sort=True)
                # Raster content can hold evidence even alongside a selectable header.
                use_ocr = not usable(text) or bool(page.get_images())
                if use_ocr:
                    text = ocr_page(page)
                if not usable(text):
                    raise IngestionError(
                        "No usable text after OCR; blank or unreadable page"
                    )
            except Exception as exc:
                raise IngestionError(f"Page {index}: {exc}") from None
            kind, document_date = page_fields(text)
            pages.append(
                {
                    "page": index,
                    "text": text,
                    "document_type": kind,
                    "document_date": document_date,
                    "extraction_method": "ocr" if use_ocr else "text",
                }
            )
        return pages


def ingestion_state(conn, claim_id):
    """Legacy registrations remain visible; new registrations require completion."""
    registered = {
        r[0]
        for r in conn.execute(
            "SELECT document_id FROM claim_documents WHERE claim_id=?", (claim_id,)
        )
    }
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='pdf_ingestions'"
    ).fetchone():
        return registered, []
    rows = conn.execute(
        "SELECT document_id,status,error FROM pdf_ingestions WHERE claim_id=?",
        (claim_id,),
    ).fetchall()
    blocked = {r[0] for r in rows if r[1] != "complete"}
    return registered - blocked, [
        {"document_id": r[0], "status": r[1], "error": r[2]}
        for r in rows
        if r[1] != "complete"
    ]


def ingest_pdf(
    pdf_path,
    claim_id,
    *,
    db_path=None,
    chroma_path=None,
    collection=None,
    embedder=None,
):
    db_path = Path(db_path or DB_PATH)
    if not db_path.is_file():
        raise IngestionError(f"Claims database does not exist: {db_path}")
    source = Path(pdf_path).expanduser().resolve()
    with sqlite3.connect(db_path, timeout=120) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        claim_id = require_claim(conn, claim_id)
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise IngestionError(
                f"Cannot read PDF: {source.name} ({type(exc).__name__})"
            ) from None
        digest = hashlib.sha256(data).hexdigest()
        document_id = f"{claim_id}-PDF-{digest}"
        ensure_schema(conn)
        conn.commit()
        # Serialize writers, including retries. Readers see pending until final commit.
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT status,page_count,chunk_count FROM pdf_ingestions WHERE document_id=?",
            (document_id,),
        ).fetchone()
        if previous and previous[0] == "complete":
            return {
                "claim_id": claim_id,
                "document_id": document_id,
                "status": "duplicate",
                "page_count": previous[1],
                "chunk_count": previous[2],
            }
        conn.execute(
            """INSERT INTO pdf_ingestions(document_id,claim_id,sha256,source,status)
            VALUES(?,?,?,?, 'pending') ON CONFLICT(document_id) DO UPDATE SET status='pending',error='' """,
            (document_id, claim_id, digest, str(source)),
        )
        # Commit pending before vector writes; interrupted work is explicitly incomplete.
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        # Another writer may have finished while waiting for this lock.
        current = conn.execute(
            "SELECT status FROM pdf_ingestions WHERE document_id=?", (document_id,)
        ).fetchone()
        if current[0] == "complete":
            return {
                "claim_id": claim_id,
                "document_id": document_id,
                "status": "duplicate",
            }
        try:
            pages = extract_pages(data)
            records = []
            for page in pages:
                meta = {k: v for k, v in page.items() if k != "text"}
                meta.update(
                    document_id=document_id,
                    claim_id=claim_id,
                    file_path=str(source),
                    filename=source.name,
                    title=source.stem,
                    sha256=digest,
                )
                for n, text in enumerate(chunk_text(page["text"]), 1):
                    records.append((f'{document_id}-P{page["page"]}-C{n}', text, meta))
            if collection is None:
                collection = chromadb.PersistentClient(
                    path=str(chroma_path or CHROMA_PATH)
                ).get_or_create_collection(
                    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
            embedder = embedder or Client(timeout=120)
            # Stable IDs + upsert allow retry after partial vector failure without duplicates.
            for start in range(0, len(records), BATCH_SIZE):
                batch = records[start : start + BATCH_SIZE]
                embeddings = embedder.embed(
                    model=EMBED_MODEL, input=[r[1] for r in batch]
                ).embeddings
                if len(embeddings) != len(batch):
                    raise IngestionError(
                        "Embedding service returned an incomplete batch"
                    )
                collection.upsert(
                    ids=[r[0] for r in batch],
                    documents=[r[1] for r in batch],
                    metadatas=[r[2] for r in batch],
                    embeddings=embeddings,
                )
            conn.execute(
                "INSERT INTO claim_documents VALUES(?,?,?,?,?,?)",
                (document_id, claim_id, "pdf", "", source.stem, str(source)),
            )
            conn.execute(
                "UPDATE pdf_ingestions SET status='complete',page_count=?,chunk_count=?,error='' WHERE document_id=?",
                (len(pages), len(records), document_id),
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            message = (
                str(exc)
                if isinstance(exc, IngestionError)
                else f"Indexing failed ({type(exc).__name__}); retry the same PDF"
            )
            conn.execute(
                "UPDATE pdf_ingestions SET status='failed',error=? WHERE document_id=?",
                (message, document_id),
            )
            conn.commit()
            raise IngestionError(message) from None
    return {
        "claim_id": claim_id,
        "document_id": document_id,
        "status": "complete",
        "page_count": len(pages),
        "chunk_count": len(records),
        "ocr_pages": [p["page"] for p in pages if p["extraction_method"] == "ocr"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path")
    parser.add_argument("--claim-id", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(ingest_pdf(args.pdf_path, args.claim_id), indent=2))
    except (IngestionError, ValueError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")


if __name__ == "__main__":
    main()

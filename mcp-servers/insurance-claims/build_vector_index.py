import sqlite3
import os
from pathlib import Path

import chromadb
import pymupdf as fitz
from ollama import Client

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("INSURANCE_DB_PATH", str(ROOT / "data/insurance_claims.db")))
CHROMA_PATH = Path(os.getenv("INSURANCE_CHROMA_PATH", str(ROOT / "data/chroma_db")))

COLLECTION_NAME = "insurance_medical_documents"
EMBED_MODEL = "embeddinggemma"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 16


def load_documents():
    """Read registered document metadata from SQLite."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                document_id,
                claim_id,
                document_type,
                document_date,
                title,
                file_path
            FROM claim_documents
            ORDER BY claim_id, document_id
        """).fetchall()

    return [dict(row) for row in rows]


def chunk_text(text):
    """Split text into overlapping character-based chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def extract_chunks(documents):
    """Extract PDF text and preserve source metadata."""

    records = []

    for document in documents:
        # Dynamic PDFs are managed by pdf_ingestion.py, including page metadata
        # and the SQLite completion gate. Do not bypass it on a legacy rebuild.
        if document["document_type"] == "pdf":
            continue
        pdf_path = ROOT / document["file_path"]

        if not pdf_path.is_file():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text")

                for chunk_index, chunk in enumerate(chunk_text(text)):
                    chunk_id = (
                        f"{document['document_id']}"
                        f"-P{page_index + 1}"
                        f"-C{chunk_index + 1}"
                    )

                    records.append(
                        {
                            "id": chunk_id,
                            "text": chunk,
                            "metadata": {
                                "document_id": document["document_id"],
                                "claim_id": document["claim_id"],
                                "document_type": document["document_type"],
                                "document_date": document["document_date"],
                                "title": document["title"],
                                "file_path": document["file_path"],
                                "page": page_index + 1,
                            },
                        }
                    )

    return records


def main():
    print("Loading SQLite document metadata...")
    documents = load_documents()

    print(f"Registered documents: {len(documents)}")

    records = extract_chunks(documents)

    if not records:
        raise RuntimeError("No searchable text extracted from PDFs.")

    print(f"Text chunks extracted: {len(records)}")

    ollama = Client()

    # Fail early if the embedding model is unavailable.
    print(f"Checking embedding model: {EMBED_MODEL}")

    test_embedding = ollama.embed(
        model=EMBED_MODEL,
        input="Insurance medical document test",
    )

    dimension = len(test_embedding.embeddings[0])
    print(f"Embedding dimensions: {dimension}")

    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Incrementally refresh registered legacy documents. Never drop a collection.
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]

        texts = [item["text"] for item in batch]

        response = ollama.embed(
            model=EMBED_MODEL,
            input=texts,
        )

        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=texts,
            embeddings=response.embeddings,
            metadatas=[item["metadata"] for item in batch],
        )

        print(
            f"Indexed {min(start + BATCH_SIZE, len(records))}" f"/{len(records)} chunks"
        )

    print("\nIndexing completed successfully.")
    print(f"ChromaDB directory: {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Indexed chunks: {collection.count()}")


if __name__ == "__main__":
    main()

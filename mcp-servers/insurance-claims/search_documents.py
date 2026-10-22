
from pathlib import Path

import chromadb
from ollama import Client


ROOT = Path(__file__).resolve().parent
CHROMA_PATH = ROOT / "data" / "chroma_db"

COLLECTION_NAME = "insurance_medical_documents"
EMBED_MODEL = "embeddinggemma"


def search_documents(question, claim_id=None, top_k=5):
    chroma = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = chroma.get_collection(
        name=COLLECTION_NAME
    )

    ollama = Client()

    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=question,
    ).embeddings[0]

    filters = (
        {"claim_id": claim_id}
        if claim_id else None
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=filters,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": text,
            "metadata": metadata,
            "distance": distance,
        })

    return matches


def main():
    print("=" * 55)
    print(" LOCAL MEDICAL DOCUMENT SEMANTIC SEARCH")
    print("=" * 55)
    print("Type 'exit' to stop.\n")

    while True:
        try:
            question = input("Search: ").strip()

            if question.lower() in ("exit", "quit"):
                break

            if not question:
                continue

            claim_id = input(
                "Claim ID (optional): "
            ).strip().upper() or None

            matches = search_documents(
                question,
                claim_id=claim_id,
            )

            if not matches:
                print("\nNo matching documents.\n")
                continue

            for index, match in enumerate(matches, 1):
                metadata = match["metadata"]

                print(f"\n--- Result {index} ---")
                print(f"Claim: {metadata['claim_id']}")
                print(f"Document: {metadata['document_id']}")
                print(f"Type: {metadata['document_type']}")
                print(f"Date: {metadata['document_date']}")
                print(f"Page: {metadata['page']}")
                print(f"Distance: {match['distance']:.4f}")
                print(f"Source: {metadata['file_path']}")
                print(f"\n{match['text']}\n")

        except (KeyboardInterrupt, EOFError):
            break

        except Exception as exc:
            print(f"\nSearch error: {exc}\n")


if __name__ == "__main__":
    main()

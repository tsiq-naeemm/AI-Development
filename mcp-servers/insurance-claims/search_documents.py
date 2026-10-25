
from pathlib import Path

import chromadb
from ollama import Client


ROOT = Path(__file__).resolve().parent
CHROMA_PATH = ROOT / "data" / "chroma_db"

COLLECTION_NAME = "insurance_medical_documents"
EMBED_MODEL = "embeddinggemma"


def search_documents(question, claim_id=None, top_k=5):
    from rag_server import search_medical_documents
    result = search_medical_documents(claim_id, question, top_k)
    if 'error' in result:
        raise ValueError(result['error'])
    return [{'text': p['text'], 'distance': p['cosine_distance'],
             'metadata': {'claim_id': p['claim_id'], 'document_id': p['document_id'],
                          'document_type': p['document_type'], 'document_date': p['document_date'],
                          'page': p['page'], 'file_path': p['source']}}
            for p in result['results']]


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
                "Claim ID (required): "
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

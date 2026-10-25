"""Run after indexing: python test_rag_server.py"""

from rag_server import search_medical_documents

CASES = [
    (
        "CLM001",
        "Find the orthopaedic referral and first physiotherapy treatment dates.",
    ),
    ("CLM003", "Was an orthopaedic referral documented before physiotherapy began?"),
    (
        "CLM004",
        "Compare the side of the ankle in the referral with the treatment record.",
    ),
    ("CLM002", "Find the specialist referral and physiotherapy discharge summary."),
]

for claim_id, question in CASES:
    result = search_medical_documents(claim_id, question)
    print(f"\n=== {claim_id} ===")
    if "error" in result:
        raise RuntimeError(result["error"])
    print("Registered documents:", result["registered_document_count"])
    for hit in result["results"]:
        print(
            f"{hit['document_id']} page {hit['page']} [{hit['document_type']}] distance={hit['cosine_distance']}"
        )
    assert result["results"], f"No retrieved evidence for {claim_id}"
print("\nAll retrieval tests passed.")


import re

import chromadb

from referral_validator import validate_anatomy


def extract_anatomy(text: str) -> str | None:
    """
    Extract a documented body side and body part.
    Supports the anatomy terms in our synthetic dataset.
    """
    match = re.search(
        r"\b(LEFT|RIGHT)\s+(knee|shoulder|ankle)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    return f"{match.group(1).upper()} {match.group(2).lower()}"


client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("insurance_medical_documents")

for claim_id in ["CLM001", "CLM002", "CLM003", "CLM004"]:
    records = collection.get(
        where={"claim_id": claim_id},
        include=["documents", "metadatas"],
    )

    referrals = []
    treatments = []

    for text, metadata in zip(
        records["documents"],
        records["metadatas"],
    ):
        anatomy = extract_anatomy(text)

        if metadata["document_type"] == "referral":
            referrals.append(anatomy)
        elif metadata["document_type"] == "treatment":
            treatments.append(anatomy)

    if not referrals:
        print(f"{claim_id}: REVIEW - No referral document available.")
        continue

    if not treatments:
        print(f"{claim_id}: REVIEW - No treatment document available.")
        continue


    if not referrals[0]:
        print(f"{claim_id}: REVIEW - Referral anatomy is not documented.")
        continue

    if not treatments[0]:
        print(f"{claim_id}: REVIEW - Treatment anatomy is not documented.")
        continue

    result = validate_anatomy(
        referrals[0],
        treatments[0],
    )

    print(
        f"{claim_id}: {result['finding']} "
        f"- {result['explanation']}"
    )

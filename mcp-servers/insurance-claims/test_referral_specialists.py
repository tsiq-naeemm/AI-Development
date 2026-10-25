import re
import chromadb

from referral_validator import validate_referring_specialist

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("insurance_medical_documents")

for claim_id in ["CLM001", "CLM002", "CLM003", "CLM004"]:
    records = collection.get(
        where={"claim_id": claim_id},
        include=["documents", "metadatas"],
    )

    referrals = [
        text
        for text, metadata in zip(
            records["documents"],
            records["metadatas"],
        )
        if metadata["document_type"] == "referral"
    ]

    if not referrals:
        print(f"{claim_id}: REVIEW - No referral document available.")
        continue

    text = " ".join(referrals)

    match = re.search(
        r"Issued by\s+(Dr\.?\s+[A-Za-z.\s]+?),\s*"
        r"(Orthopaedic Specialist|Orthopedic Specialist)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        name = match.group(1).strip()
        specialty = match.group(2).strip()
    else:
        name = ""
        specialty = ""

    result = validate_referring_specialist(name, specialty)

    print(f"{claim_id}: {result['finding']} " f"- {result['explanation']}")

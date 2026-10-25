import chromadb

from referral_validator import validate_referral_date

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("insurance_medical_documents")

for claim_id in ["CLM001", "CLM002", "CLM003", "CLM004"]:
    records = collection.get(
        where={"claim_id": claim_id},
        include=["metadatas"],
    )

    referrals = [
        meta for meta in records["metadatas"] if meta["document_type"] == "referral"
    ]

    treatments = [
        meta for meta in records["metadatas"] if meta["document_type"] == "treatment"
    ]

    if not referrals:
        print(f"{claim_id}: REVIEW - No referral document available.")
        continue

    if not treatments:
        print(f"{claim_id}: REVIEW - No treatment document available.")
        continue

    for referral in referrals:
        for treatment in treatments:
            result = validate_referral_date(
                referral["document_date"],
                treatment["document_date"],
            )

            print(f"{claim_id}: {result['finding']} " f"- {result['explanation']}")

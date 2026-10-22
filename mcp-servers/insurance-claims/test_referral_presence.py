
import chromadb

from referral_validator import validate_referral_presence


client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("insurance_medical_documents")

for claim_id in ["CLM001", "CLM002", "CLM003", "CLM004"]:
    records = collection.get(
        where={"claim_id": claim_id},
        include=["metadatas"],
    )

    document_types = [
        metadata["document_type"]
        for metadata in records["metadatas"]
    ]

    result = validate_referral_presence(document_types)

    print(
        f"{claim_id}: {result['finding']} "
        f"- {result['explanation']}"
    )


from datetime import date


def validate_referral_date(
    referral_date: str,
    treatment_date: str,
) -> dict:
    """
    Check whether a referral predates treatment.

    Dates must be ISO formatted: YYYY-MM-DD.
    """

    referral = date.fromisoformat(referral_date)
    treatment = date.fromisoformat(treatment_date)

    if referral < treatment:
        finding = "PASS"
        explanation = "Referral predates treatment."

    elif referral == treatment:
        finding = "REVIEW"
        explanation = (
            "Referral and treatment have the same date. "
            "Their sequence cannot be established from dates alone."
        )

    else:
        finding = "FAIL"
        explanation = "Referral is dated after treatment."

    return {
        "referral_date": referral_date,
        "treatment_date": treatment_date,
        "finding": finding,
        "explanation": explanation,
    }

def validate_anatomy(
    referral_body_part: str,
    treatment_body_part: str,
) -> dict:
    """
    Compare the body part and side specified in the
    referral and treatment records.
    """
    referral = " ".join(referral_body_part.lower().split())
    treatment = " ".join(treatment_body_part.lower().split())

    if not referral or not treatment:
        finding = "REVIEW"
        explanation = "Body part information is missing."

    elif referral == treatment:
        finding = "PASS"
        explanation = "Referral and treatment anatomy match."

    else:
        finding = "REVIEW"
        explanation = (
            "Referral and treatment anatomy differ. "
            "Manual clarification is required."
        )

    return {
        "referral_anatomy": referral_body_part,
        "treatment_anatomy": treatment_body_part,
        "finding": finding,
        "explanation": explanation,
    }

def validate_referring_specialist(
    specialist_name: str,
    specialist_specialty: str,
) -> dict:
    """
    Check whether the referral identifies a named
    orthopaedic specialist.
    """
    name = " ".join(specialist_name.split())
    specialty = " ".join(specialist_specialty.lower().split())

    if not name:
        finding = "REVIEW"
        explanation = "Referring specialist's name is missing."

    elif not specialty:
        finding = "REVIEW"
        explanation = "Referring specialist's specialty is missing."

    elif specialty in ("orthopaedic specialist", "orthopedic specialist"):
        finding = "PASS"
        explanation = "Named orthopaedic specialist documented."

    else:
        finding = "REVIEW"
        explanation = "Referring clinician's specialty requires verification."

    return {
        "specialist_name": name,
        "specialist_specialty": specialist_specialty,
        "finding": finding,
        "explanation": explanation,
    }

def validate_referral_presence(document_types: list[str]) -> dict:
    """
    Check whether the available claim documents
    include a referral.
    """
    normalized_types = {
        document_type.strip().lower()
        for document_type in document_types
    }

    if "referral" in normalized_types:
        finding = "PASS"
        explanation = "Referral document is present."
    else:
        finding = "REVIEW"
        explanation = (
            "No referral document found in the available records. "
            "This does not prove that no referral exists elsewhere."
        )

    return {
        "finding": finding,
        "explanation": explanation,
    }


def combine_validation_results(checks: dict) -> dict:
    """
    Combine individual validation findings.

    FAIL takes precedence over REVIEW.
    REVIEW takes precedence over PASS.
    """
    findings = [result["finding"] for result in checks.values()]

    if not findings:
        overall = "REVIEW"
    elif "FAIL" in findings:
        overall = "FAIL"
    elif "REVIEW" in findings:
        overall = "REVIEW"
    elif all(finding == "PASS" for finding in findings):
        overall = "PASS"
    else:
        overall = "REVIEW"

    return {
        "overall_finding": overall,
        "checks": checks,
    }

if __name__ == "__main__":
    tests = [
        ("CLM001", "2026-01-05", "2026-01-09"),
        ("CLM002", "2026-02-02", "2026-02-06"),
        ("CLM004", "2026-04-03", "2026-04-06"),
        ("SAME-DAY", "2026-04-06", "2026-04-06"),
        ("LATE-REFERRAL", "2026-04-10", "2026-04-06"),
    ]

    for claim_id, referral_date, treatment_date in tests:
        result = validate_referral_date(
            referral_date,
            treatment_date,
        )

        print(
            f"{claim_id}: {result['finding']} "
            f"- {result['explanation']}"
        )

    
    print("\nAnatomy validation:")

    anatomy_tests = [
        ("CLM001", "RIGHT knee", "RIGHT knee"),
        ("CLM002", "LEFT shoulder", "LEFT shoulder"),
        ("CLM004", "LEFT ankle", "RIGHT ankle"),
    ]

    for claim_id, referral_anatomy, treatment_anatomy in anatomy_tests:
        result = validate_anatomy(
            referral_anatomy,
            treatment_anatomy,
        )
        print(
            f"{claim_id}: {result['finding']} "
            f"- {result['explanation']}"
        )

    
    print("\nSpecialist validation:")

    specialist_tests = [
        ("CLM001", "Dr A Malik", "Orthopaedic Specialist"),
        ("CLM002", "Dr N Qureshi", "Orthopaedic Specialist"),
        ("CLM004", "Dr S Farooq", "Orthopaedic Specialist"),
        ("MISSING-NAME", "", "Orthopaedic Specialist"),
        ("UNKNOWN-SPECIALTY", "Dr X", ""),
    ]

    for claim_id, name, specialty in specialist_tests:
        result = validate_referring_specialist(name, specialty)
        print(
            f"{claim_id}: {result['finding']} "
            f"- {result['explanation']}"
        )



    print("\nReferral presence validation:")

    presence_tests = [
        ("CLM001", ["consultation", "referral", "treatment", "claim_form"]),
        ("CLM003", ["consultation", "treatment", "claim_form"]),
        ("CLM004", ["consultation", "referral", "treatment", "claim_form"]),
    ]

    for claim_id, document_types in presence_tests:
        result = validate_referral_presence(document_types)
        print(
            f"{claim_id}: {result['finding']} "
            f"- {result['explanation']}"
        )


    print("\nCombined validation:")

    test_cases = {
        "ALL-PASS": ["PASS", "PASS", "PASS", "PASS"],
        "MISSING-EVIDENCE": ["PASS", "PASS", "REVIEW", "PASS"],
        "DATE-FAILURE": ["PASS", "FAIL", "PASS", "PASS"],
    }

    for case_name, findings in test_cases.items():
        checks = {
            f"check_{i}": {"finding": finding}
            for i, finding in enumerate(findings, start=1)
        }

        result = combine_validation_results(checks)

        print(f"{case_name}: {result['overall_finding']}")

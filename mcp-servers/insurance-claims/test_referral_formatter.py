from langgraph_rag_agent import format_referral_validation


def test_clm001():
    result = {
        "claim_id": "CLM001",
        "overall_finding": "PASS",
        "checks": {
            "referral_presence": {
                "finding": "PASS",
                "explanation": "Referral document is present."
            },
            "referral_date": {
                "finding": "PASS",
                "explanation": "Referral predates treatment.",
                "referral_date": "2026-01-05",
                "treatment_date": "2026-01-09",
                "source_document_ids": [
                    "CLM001-DOC02",
                    "CLM001-DOC03"
                ]
            },
            "anatomy": {
                "finding": "PASS",
                "explanation": "Referral and treatment anatomy match.",
                "referral_anatomy": "RIGHT knee",
                "treatment_anatomy": "RIGHT knee"
            },
            "specialist": {
                "finding": "PASS",
                "explanation": "Named orthopaedic specialist documented.",
                "specialist_name": "Dr. A. Malik",
                "specialist_specialty": "Orthopaedic Specialist"
            }
        },
        "note": "Evidence validation only; not an insurance approval or denial."
    }

    report = format_referral_validation(result)

    assert "Overall finding: PASS" in report
    assert "2026-01-05" in report
    assert "2026-01-09" in report
    assert "RIGHT knee" in report
    assert "Dr. A. Malik" in report
    assert "Supporting documents: CLM001-DOC02, CLM001-DOC03" in report
    assert (
        "Conclusion: The available evidence supports the "
        "specialist referral validation."
    ) in report
    assert "Note: Evidence validation only" in report

    print("PASS: CLM001 formatter test")

def test_clm002():
    result = {
        "claim_id": "CLM002",
        "overall_finding": "REVIEW",
        "checks": {
            "referral_presence": {
                "finding": "PASS",
                "explanation": "Referral document is present."
            },
            "referral_date": {
                "finding": "PASS",
                "explanation": "Referral predates treatment."
            },
            "anatomy": {
                "finding": "REVIEW",
                "explanation": "Body part information is missing.",
                "referral_anatomy": "LEFT shoulder",
                "treatment_anatomy": "",
                "source_document_ids": [
                    "CLM002-DOC02",
                    "CLM002-DOC03"
                ]
            },
            "specialist": {
                "finding": "PASS",
                "explanation": "Named orthopaedic specialist documented."
            }
        },
        "note": "Evidence validation only; not an insurance approval or denial."
    }

    report = format_referral_validation(result)

    assert "Claim: CLM002" in report
    assert "Overall finding: REVIEW" in report
    assert "Anatomy: REVIEW - Body part information is missing." in report
    assert "Reason: Anatomy: Body part information is missing." in report
    assert "Supporting documents: CLM002-DOC02, CLM002-DOC03" in report
    assert "Conclusion: The available evidence requires manual review." in report
    assert "Note: Evidence validation only" in report

    print("PASS: CLM002 formatter test")

def test_clm003():
    result = {
        "claim_id": "CLM003",
        "overall_finding": "REVIEW",
        "checks": {
            "referral_presence": {
                "finding": "REVIEW",
                "explanation": (
                    "No referral document found in the available records. "
                    "This does not prove that no referral exists elsewhere."
                )
            },
            "referral_date": {
                "finding": "REVIEW",
                "explanation": "No referral available for date comparison."
            },
            "anatomy": {
                "finding": "REVIEW",
                "explanation": "No referral available for anatomy comparison."
            },
            "specialist": {
                "finding": "REVIEW",
                "explanation": "No referral available for specialist verification."
            }
        },
        "source_document_ids": [
            "CLM003-DOC01",
            "CLM003-DOC02",
            "CLM003-DOC03"
        ],
        "note": (
            "Evidence validation only; not an insurance approval or denial. "
            "Missing evidence may exist elsewhere."
        )
    }

    report = format_referral_validation(result)

    assert "Claim: CLM003" in report
    assert "Overall finding: REVIEW" in report
    assert "Referral presence: REVIEW" in report
    assert "Referral date: REVIEW" in report
    assert "Anatomy: REVIEW" in report
    assert "Specialist: REVIEW" in report
    assert "does not prove that no referral exists elsewhere" in report
    assert "No referral available for date comparison" in report
    assert "Supporting documents: CLM003-DOC01, CLM003-DOC02, CLM003-DOC03" in report
    assert "Conclusion: The available evidence requires manual review." in report
    assert "Note: Evidence validation only" in report

    print("PASS: CLM003 formatter test")


def test_clm004():
    result = {
        "claim_id": "CLM004",
        "overall_finding": "REVIEW",
        "checks": {
            "referral_presence": {
                "finding": "PASS",
                "explanation": "Referral document is present."
            },
            "referral_date": {
                "finding": "PASS",
                "explanation": "Referral predates treatment."
            },
            "anatomy": {
                "finding": "REVIEW",
                "explanation": (
                    "Referral and treatment anatomy differ. "
                    "Manual clarification is required."
                ),
                "referral_anatomy": "LEFT ankle",
                "treatment_anatomy": "RIGHT ankle",
                "source_document_ids": [
                    "CLM004-DOC02",
                    "CLM004-DOC03"
                ]
            },
            "specialist": {
                "finding": "PASS",
                "explanation": "Named orthopaedic specialist documented."
            }
        },
        "note": (
            "Evidence validation only; not an insurance approval or denial. "
            "Missing evidence may exist elsewhere."
        )
    }

    report = format_referral_validation(result)

    assert "Claim: CLM004" in report
    assert "Overall finding: REVIEW" in report
    assert "Referral presence: PASS" in report
    assert "Referral date: PASS" in report
    assert "Anatomy: REVIEW" in report
    assert "Specialist: PASS" in report

    assert (
        "referral specifies LEFT ankle but treatment specifies RIGHT ankle"
        in report
    )

    assert (
        "Supporting documents: CLM004-DOC02, CLM004-DOC03"
        in report
    )

    assert (
        "Conclusion: The available evidence requires manual review."
        in report
    )

    assert "Note: Evidence validation only" in report

    print("PASS: CLM004 formatter test")


if __name__ == "__main__":
    test_clm001()
    test_clm002()
    test_clm003()
    test_clm004()

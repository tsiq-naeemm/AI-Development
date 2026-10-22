
from referral_validator import combine_validation_results


claim_checks = {
    "CLM001": {
        "referral_presence": "PASS",
        "referral_date": "PASS",
        "anatomy": "PASS",
        "specialist": "PASS",
    },
    "CLM002": {
        "referral_presence": "PASS",
        "referral_date": "PASS",
        "anatomy": "REVIEW",
        "specialist": "PASS",
    },
    "CLM003": {
        "referral_presence": "REVIEW",
        "referral_date": "REVIEW",
        "anatomy": "REVIEW",
        "specialist": "REVIEW",
    },
    "CLM004": {
        "referral_presence": "PASS",
        "referral_date": "PASS",
        "anatomy": "REVIEW",
        "specialist": "PASS",
    },
}

for claim_id, findings in claim_checks.items():
    checks = {
        check_name: {"finding": finding}
        for check_name, finding in findings.items()
    }

    result = combine_validation_results(checks)

    print(f"{claim_id}: {result['overall_finding']}")

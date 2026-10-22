
from mcp.server import MCPServer

# Initialize the MCP server
mcp = MCPServer("InsuranceClaims")

# Synthetic insurance claims
CLAIMS = {
    "CLM001": {"status": "Open", "amount": 125000},
    "CLM002": {"status": "Closed", "amount": 45000},
    "CLM003": {"status": "Under Investigation", "amount": 210000},
    "CLM004": {"status": "Open", "amount": 78000},
}


@mcp.tool()
def get_claim_status(claim_id: str) -> str:
    """Get the current status of an insurance claim."""
    claim = CLAIMS.get(claim_id)
    return claim["status"] if claim else "Claim not found"


@mcp.tool()
def get_claim_amount(claim_id: str) -> dict:
    """Get the financial amount of an insurance claim."""
    claim = CLAIMS.get(claim_id)

    if not claim:
        return {"error": "Claim not found"}

    return {
        "claim_id": claim_id,
        "amount": claim["amount"],
        "currency": "USD",
    }


@mcp.tool()
def list_open_claims() -> list[dict]:
    """List all currently open insurance claims."""
    return [
        {"claim_id": claim_id, **claim}
        for claim_id, claim in CLAIMS.items()
        if claim["status"] == "Open"
    ]


if __name__ == "__main__":
    mcp.run()

import sqlite3
from pathlib import Path

from mcp.server import MCPServer

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "insurance_claims.db"

server = MCPServer("InsuranceClaimsSQLite")


def get_connection():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. " "Run python init_database.py first."
        )

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@server.tool()
def get_claim_status(claim_id: str) -> str:
    """Return the current status of an insurance claim."""

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT status
            FROM claims
            WHERE claim_id = ?
            """,
            (claim_id,),
        ).fetchone()

    if row is None:
        return "Claim not found"

    return row["status"]


@server.tool()
def get_claim_amount(claim_id: str) -> dict:
    """Return the amount and currency of a claim."""

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT claim_id, amount, currency
            FROM claims
            WHERE claim_id = ?
            """,
            (claim_id,),
        ).fetchone()

    if row is None:
        return {"error": "Claim not found"}

    return {
        "claim_id": row["claim_id"],
        "amount": row["amount"],
        "currency": row["currency"],
    }


@server.tool()
def list_open_claims() -> list[dict]:
    """List all open claims with their amounts."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT claim_id, status, amount
            FROM claims
            WHERE status = ?
            ORDER BY claim_id
            """,
            ("Open",),
        ).fetchall()

    return [
        {
            "claim_id": row["claim_id"],
            "status": row["status"],
            "amount": row["amount"],
        }
        for row in rows
    ]


if __name__ == "__main__":
    server.run()

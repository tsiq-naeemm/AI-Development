
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "insurance_claims.db"

CLAIMS = [
    ("CLM001", "Open", 125000, "USD"),
    ("CLM002", "Closed", 45000, "USD"),
    ("CLM003", "Under Investigation", 210000, "USD"),
    ("CLM004", "Open", 78000, "USD"),
]


def initialize_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                amount INTEGER NOT NULL
                    CHECK (amount >= 0),
                currency TEXT NOT NULL
                    DEFAULT 'USD'
            )
        """)

        conn.executemany("""
            INSERT OR IGNORE INTO claims
                (claim_id, status, amount, currency)
            VALUES (?, ?, ?, ?)
        """, CLAIMS)

        conn.commit()

        rows = conn.execute("""
            SELECT claim_id, status, amount, currency
            FROM claims
            ORDER BY claim_id
        """).fetchall()

    print(f"Database: {DB_PATH}")
    print(f"Total claims: {len(rows)}")

    for row in rows:
        print(row)


if __name__ == "__main__":
    initialize_database()

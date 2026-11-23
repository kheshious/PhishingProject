
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app import get_db_connection


def upgrade_gmail_tracking():
    conn = get_db_connection()
    cursor = conn.cursor()

    columns = {
        "sender": "TEXT NULL",
        "subject": "TEXT NULL",
        "email_date": "VARCHAR(255) NULL",
        "phishing_probability": "FLOAT NULL",
        "legitimate_probability": "FLOAT NULL",
        "recommended_actions": "TEXT NULL",
        "viewed_at": "DATETIME NULL"
    }

    try:
        cursor.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'gmail_detection_tracking'
            """
        )

        existing = {row[0] for row in cursor.fetchall()}

        if not existing:
            raise RuntimeError(
                "gmail_detection_tracking table was not found."
            )

        for column, definition in columns.items():
            if column not in existing:
                cursor.execute(
                    f"ALTER TABLE gmail_detection_tracking "
                    f"ADD COLUMN {column} {definition}"
                )
                print(f"Added: {column}")
            else:
                print(f"Already exists: {column}")

        conn.commit()

        print("\nGmail tracking upgrade completed.")
        print("Existing detection history was preserved.")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    upgrade_gmail_tracking()


import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app import get_db_connection


PHISHAI_USER_ID = 1


def save_gmail_detection(
    gmail_message_id,
    email_body,
    prediction_result,
    sender=None,
    subject=None,
    email_date=None,
):
    """
    Save Gmail detection to History and register
    a pending live email notification.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Prevent duplicate Gmail detections.
        try:
            cursor.execute(
                """
                INSERT INTO gmail_detection_tracking
                    (gmail_message_id, user_id, detection_id)
                VALUES (%s, %s, %s)
                """,
                (
                    gmail_message_id,
                    PHISHAI_USER_ID,
                    0,
                ),
            )

        except Exception as error:
            if getattr(error, "errno", None) == 1062:
                conn.rollback()
                return False

            raise

        # Save detection to existing History table.
        cursor.execute(
            """
            INSERT INTO detections
            (
                user_id,
                input_type,
                submitted_content,
                prediction,
                confidence_score,
                risk_level,
                explanation
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                PHISHAI_USER_ID,
                "Email",
                email_body,
                prediction_result["prediction"],
                float(prediction_result["confidence"]),
                prediction_result["risk_level"],
                "Automatic Gmail scan. " + "; ".join(
                    prediction_result.get("reasons", [])
                ),
            ),
        )

        detection_id = cursor.lastrowid

        phishing_probability = prediction_result.get(
            "phishing_probability"
        )

        legitimate_probability = prediction_result.get(
            "legitimate_probability"
        )

        recommended_actions = prediction_result.get(
            "recommended_actions", []
        )

        # Save additional live notification information.
        cursor.execute(
            """
            UPDATE gmail_detection_tracking
            SET
                detection_id = %s,
                sender = %s,
                subject = %s,
                email_date = %s,
                phishing_probability = %s,
                legitimate_probability = %s,
                recommended_actions = %s,
                viewed_at = NULL
            WHERE gmail_message_id = %s
              AND user_id = %s
            """,
            (
                detection_id,
                sender,
                subject,
                email_date,
                float(phishing_probability)
                if phishing_probability is not None
                else None,
                float(legitimate_probability)
                if legitimate_probability is not None
                else None,
                json.dumps(recommended_actions),
                gmail_message_id,
                PHISHAI_USER_ID,
            ),
        )

        conn.commit()

        print(
            f"Gmail detection saved to PhishAI history. "
            f"Detection ID: {detection_id}"
        )

        print("Gmail notification registered.")

        return True

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()

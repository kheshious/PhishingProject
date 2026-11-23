
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from gmail_scanner import extract_email_body, predict_email
from gmail_database import save_gmail_detection


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TOKEN_FILE = BASE_DIR / "token.json"
STATE_FILE = BASE_DIR / "gmail_monitor_state.json"
STATUS_FILE = BASE_DIR / "gmail_monitor_status.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]

CHECK_INTERVAL = 30
MAX_RESULTS = 50


# ============================================================
# LOAD PREVIOUSLY PROCESSED EMAILS
# ============================================================

def load_seen_ids():

    if STATE_FILE.exists():

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return set(json.load(file))

    return set()


# ============================================================
# SAVE PROCESSED EMAIL IDS
# ============================================================

def save_seen_ids(seen_ids):

    temporary_file = STATE_FILE.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            sorted(seen_ids),
            file,
            indent=2
        )

    temporary_file.replace(STATE_FILE)


# ============================================================
# GMAIL AUTHENTICATION
# ============================================================

def get_service():

    creds = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        SCOPES
    )

    if not creds.valid:

        if creds.expired and creds.refresh_token:

            creds.refresh(Request())

            TOKEN_FILE.write_text(
                creds.to_json(),
                encoding="utf-8"
            )

        else:

            raise RuntimeError(
                "Gmail authentication expired. "
                "Run gmail_auth.py again."
            )

    return build(
        "gmail",
        "v1",
        credentials=creds
    )


# ============================================================
# MONITORING STATUS FOR FLASK WEBSITE
# ============================================================

def update_monitor_status(account, next_check_at=None):

    now = datetime.now(timezone.utc)

    status_data = {
        "account": account,
        "updated_at": now.isoformat(),
        "next_check_at": (
            next_check_at.isoformat()
            if next_check_at is not None
            else None
        )
    }

    temporary_file = STATUS_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            status_data,
            indent=2
        ),
        encoding="utf-8"
    )

    temporary_file.replace(STATUS_FILE)


# ============================================================
# SCAN NEW EMAIL
# ============================================================

def scan_new_email(service, message_id):

    email = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute()

    headers = {
        header["name"].lower(): header["value"]
        for header in email.get(
            "payload", {}
        ).get("headers", [])
    }

    sender = headers.get(
        "from",
        "Unknown"
    )

    subject = headers.get(
        "subject",
        "No subject"
    )

    date = headers.get(
        "date",
        "Unknown"
    )

    body = extract_email_body(
        email.get("payload", {})
    )

    print("\n" + "=" * 65)
    print("NEW EMAIL DETECTED")
    print("=" * 65)

    print("From:", sender)
    print("Subject:", subject)
    print("Date:", date)

    print("\n--- EMAIL BODY ---")
    print("Body length:", len(body))
    print(
        "Body preview:",
        body[:300] if body else "[EMPTY BODY]"
    )
    print("--- END BODY ---")

    # IMPORTANT:
    # Only the email body is classified.
    # Sender and subject are not model inputs.

    if not body.strip():

        print("\nSKIPPED: Email body is empty.")
        print("No prediction was made.")
        print("=" * 65)

        return True

    # ========================================================
    # AI PREDICTION
    # ========================================================

    result = predict_email(body)

    # ========================================================
    # SAVE DETECTION TO EXISTING PHISHAI DATABASE
    # ========================================================

    saved = save_gmail_detection(
    gmail_message_id=message_id,
    email_body=body,
    prediction_result=result,
    sender=sender,
    subject=subject,
    email_date=date
    )

    if saved:

        print(
            "\nDATABASE: Detection saved "
            "to PhishAI History."
        )

    else:

        print(
            "\nDATABASE: Email already saved. "
            "Skipping duplicate."
        )

    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    print("\n--- PHISHAI PREDICTION ---")

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Model confidence:",
        f'{result["confidence"]}%'
    )

    print(
        "Phishing risk:",
        f'{result["risk_score"]}%'
    )

    print(
        "Risk level:",
        result["risk_level"]
    )

    print(
        "Legitimate probability:",
        f'{result["legitimate_probability"]}%'
    )

    print(
        "Phishing probability:",
        f'{result["phishing_probability"]}%'
    )

    if result["prediction"] == "Phishing":

        print(
            "\nWARNING: Potential phishing "
            "email detected!"
        )

        print(
            "Please verify the email "
            "before taking action."
        )

    print("=" * 65)

    return True


# ============================================================
# BACKGROUND GMAIL MONITOR
# ============================================================

def monitor_gmail():

    service = get_service()

    seen_ids = load_seen_ids()

    # Get the actual Gmail account connected via OAuth.

    profile = service.users().getProfile(
        userId="me"
    ).execute()

    account = profile.get(
        "emailAddress",
        "Unknown Gmail account"
    )

    print("\n" + "=" * 65)
    print("PHISHAI LIVE GMAIL MONITOR")
    print("=" * 65)

    print("Connected Gmail account:", account)

    print(
        "Classification input: EMAIL BODY ONLY"
    )

    print(
        f"Checking for new emails every "
        f"{CHECK_INTERVAL} seconds."
    )

    print(
        "Live monitoring status enabled."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print("=" * 65)

    # Publish status immediately so the website
    # does not have to wait for the first scan.

    update_monitor_status(
        account=account,
        next_check_at=datetime.now(timezone.utc)
    )

    try:

        while True:

            try:

                response = service.users().messages().list(
                    userId="me",
                    labelIds=["INBOX"],
                    maxResults=MAX_RESULTS
                ).execute()

                messages = response.get(
                    "messages",
                    []
                )

                # First-time setup:
                # Treat existing inbox emails as the baseline.
                # Only future emails will be scanned.

                if not STATE_FILE.exists():

                    seen_ids.update(
                        message["id"]
                        for message in messages
                    )

                    save_seen_ids(seen_ids)

                    print(
                        "\nInitial inbox baseline saved."
                    )

                else:

                    new_messages = [
                        message
                        for message in reversed(messages)
                        if message["id"] not in seen_ids
                    ]

                    for message in new_messages:

                        message_id = message["id"]

                        try:

                            success = scan_new_email(
                                service,
                                message_id
                            )

                            # Mark as processed only after
                            # successful scanning and saving.

                            if success:

                                seen_ids.add(
                                    message_id
                                )

                                save_seen_ids(
                                    seen_ids
                                )

                        except Exception as error:

                            print(
                                "\nFailed to scan email:",
                                message_id,
                                error
                            )

            except Exception as error:

                print(
                    "\nGmail monitoring error:",
                    error
                )

            # =================================================
            # UPDATE WEBSITE STATUS AND COUNTDOWN
            # =================================================

            next_check = (
                datetime.now(timezone.utc)
                + timedelta(
                    seconds=CHECK_INTERVAL
                )
            )

            update_monitor_status(
                account=account,
                next_check_at=next_check
            )

            print(
                "\nNext Gmail check:",
                next_check.astimezone().strftime(
                    "%H:%M:%S"
                )
            )

            # Sleep in short intervals, refreshing the
            # heartbeat so Flask can confirm the monitor
            # is still running.

            while True:

                remaining = (
                    next_check
                    - datetime.now(timezone.utc)
                ).total_seconds()

                if remaining <= 0:
                    break

                time.sleep(
                    min(5, remaining)
                )

                update_monitor_status(
                    account=account,
                    next_check_at=next_check
                )

    except KeyboardInterrupt:

        print(
            "\nPhishAI Gmail monitoring stopped."
        )

    finally:

        # Remove the heartbeat when the monitor stops.
        # This prevents Flask from displaying a stale
        # Monitoring status.

        try:

            STATUS_FILE.unlink(
                missing_ok=True
            )

        except OSError:
            pass


# ============================================================
# START MONITOR
# ============================================================

if __name__ == "__main__":

    monitor_gmail()

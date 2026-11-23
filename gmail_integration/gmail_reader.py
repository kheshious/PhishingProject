
import base64
from pathlib import Path

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

BASE_DIR = Path(__file__).resolve().parent
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def extract_email_body(payload):
    """Extract plain-text email content, including nested MIME parts."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(
                data + "=" * (-len(data) % 4)
            ).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        body = extract_email_body(part)
        if body:
            return body

    return ""


def read_emails():
    creds = Credentials.from_authorized_user_file(
        str(TOKEN_FILE), SCOPES
    )

    service = build("gmail", "v1", credentials=creds)

    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=5
    ).execute()

    messages = results.get("messages", [])

    if not messages:
        print("No emails found.")
        return

    print(f"Found {len(messages)} emails:\n")

    for message in messages:
        email = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()

        headers = {
            h["name"].lower(): h["value"]
            for h in email.get("payload", {}).get("headers", [])
        }

        body = extract_email_body(email.get("payload", {}))

        print("From:", headers.get("from", "Unknown"))
        print("Subject:", headers.get("subject", "No subject"))
        print("Date:", headers.get("date", "Unknown"))
        print("Body preview:", body[:200] if body else "No plain-text body")
        print("-" * 50)


if __name__ == "__main__":
    read_emails()

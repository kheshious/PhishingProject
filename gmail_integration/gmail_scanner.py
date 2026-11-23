
import base64
import html
import re
import sys
from pathlib import Path
from html.parser import HTMLParser

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TOKEN_FILE = BASE_DIR / "token.json"

sys.path.insert(0, str(PROJECT_DIR))

from machine_learning.predict_email import predict_email

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class EmailHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

        if tag in ("p", "br", "div", "li"):
            self.text.append(" ")

        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.text.append(" " + value + " ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)

    def get_text(self):
        return html.unescape(" ".join(self.text))


def decode_body(data):
    if not data:
        return ""

    decoded = base64.urlsafe_b64decode(
        data + "=" * (-len(data) % 4)
    )
    return decoded.decode("utf-8", errors="replace")


def extract_email_body(payload):
    plain_parts = []
    html_parts = []

    def walk(part):
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")

        if part.get("filename"):
            return

        if mime_type == "text/plain" and data:
            plain_parts.append(decode_body(data))

        elif mime_type == "text/html" and data:
            parser = EmailHTMLParser()
            parser.feed(decode_body(data))
            html_parts.append(parser.get_text())

        for child in part.get("parts", []):
            walk(child)

    walk(payload)

    if plain_parts:
        return "\n".join(plain_parts)

    if html_parts:
        return "\n".join(html_parts)

    return ""


def scan_gmail():
    creds = Credentials.from_authorized_user_file(
        str(TOKEN_FILE), SCOPES
    )

    service = build("gmail", "v1", credentials=creds)

    response = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=5
    ).execute()

    messages = response.get("messages", [])

    print(f"\nScanning {len(messages)} Gmail messages...")

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

        sender = headers.get("from", "Unknown")
        subject = headers.get("subject", "No subject")
        body = extract_email_body(email.get("payload", {}))

        email_text = (
            f"Subject: {subject}\n"
            f"From: {sender}\n\n"
            f"{body}"
        )

        if not email_text.strip():
            print("Skipping empty email.")
            continue

        result = predict_email(email_text)

        print("\n" + "=" * 55)
        print("Subject:", subject)
        print("Prediction:", result["prediction"])
        print("Confidence:", f'{result["confidence"]}%')
        print("Phishing risk:", f'{result["risk_score"]}%')
        print("Risk level:", result["risk_level"])

        if result["prediction"] == "Phishing":
            print("WARNING: Potential phishing email detected!")

    print("\nGmail scanning completed.")


if __name__ == "__main__":
    scan_gmail()

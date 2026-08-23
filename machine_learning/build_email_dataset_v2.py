import html
import mailbox
import re
import os
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd


# =========================================================
# File paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

LEGITIMATE_DATASET_PATH = DATASETS_DIR / "phishing_email.csv"

NAZARIO_FILES = [
    DATASETS_DIR / "phishing-2023.txt",
    DATASETS_DIR / "phishing-2024.txt",
    DATASETS_DIR / "phishing-2025.txt",
]
OUTPUT_PATH = DATASETS_DIR / "email_training_v2.csv"


# =========================================================
# Dataset limits
# =========================================================

# The Enron corpus is extremely large.
# We first collect phishing emails, then select the same
# number of legitimate emails to create a balanced dataset.

MAX_LEGITIMATE_EMAILS = 15000
RANDOM_STATE = 42


# =========================================================
# Text cleaning helpers
# =========================================================

def clean_email_text(value):
    """
    Clean email text while preserving meaningful words.

    This cleaning is only for building the dataset.
    The final training script can perform further preprocessing.
    """

    if value is None:
        return ""

    text = str(value)

    text = html.unescape(text)

    # Remove HTML tags.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove excessive whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def load_legitimate_emails(limit):


    print("\nLoading legitimate emails...")

    if not LEGITIMATE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Legitimate dataset not found: {LEGITIMATE_DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        LEGITIMATE_DATASET_PATH
    )

    legitimate_dataframe = dataframe[
        dataframe["label"] == 0
    ].copy()

    legitimate_dataframe["text_combined"] = (
        legitimate_dataframe["text_combined"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    legitimate_dataframe = legitimate_dataframe[
        legitimate_dataframe["text_combined"].str.len() >= 30
    ]

    legitimate_dataframe = legitimate_dataframe.drop_duplicates(
        subset=["text_combined"]
    )

    if len(legitimate_dataframe) > limit:
        legitimate_dataframe = legitimate_dataframe.sample(
            n=limit,
            random_state=RANDOM_STATE
        )

    records = []

    for _, row in legitimate_dataframe.iterrows():

        text = row["text_combined"]

        records.append({
            "subject": "",
            "sender": "",
            "receiver": "",
            "body": text,
            "text_combined": text,
            "url_count": count_urls(text),
            "attachment_count": 0,
            "message_length": len(text),
            "label": 0,
            "source": "Existing-Legitimate-Corpus",
        })

    print(
        f"Legitimate emails collected: {len(records):,}"
    )

    return records


def decode_payload(payload, charset=None):
    """
    Decode raw email payload safely.
    """

    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    encodings_to_try = []

    if charset:
        encodings_to_try.append(charset)

    encodings_to_try.extend([
        "utf-8",
        "latin-1",
        "windows-1252",
    ])

    for encoding in encodings_to_try:
        try:
            return payload.decode(
                encoding,
                errors="replace",
            )
        except (
            LookupError,
            UnicodeDecodeError,
            AttributeError,
        ):
            continue

    return str(payload)


def extract_email_body(message):
    """
    Extract the readable body from a parsed email.

    Plain text is preferred. HTML is used when plain text
    is unavailable.
    """

    plain_parts = []
    html_parts = []

    if message.is_multipart():

        for part in message.walk():

            content_type = part.get_content_type()
            disposition = str(
                part.get_content_disposition() or ""
            ).lower()

            # Do not put attachment content into the email body.
            if disposition == "attachment":
                continue

            try:
                payload = part.get_payload(decode=True)
            except Exception:
                payload = None

            decoded_text = decode_payload(
                payload,
                part.get_content_charset(),
            )

            if content_type == "text/plain":
                plain_parts.append(decoded_text)

            elif content_type == "text/html":
                html_parts.append(decoded_text)

    else:

        try:
            payload = message.get_payload(decode=True)
        except Exception:
            payload = None

        decoded_text = decode_payload(
            payload,
            message.get_content_charset(),
        )

        if message.get_content_type() == "text/html":
            html_parts.append(decoded_text)
        else:
            plain_parts.append(decoded_text)

    if plain_parts:
        body = " ".join(plain_parts)
    else:
        body = " ".join(html_parts)

    return clean_email_text(body)


def count_attachments(message):
    """
    Count email attachments.
    """

    attachment_count = 0

    if not message.is_multipart():
        return attachment_count

    for part in message.walk():

        disposition = str(
            part.get_content_disposition() or ""
        ).lower()

        filename = part.get_filename()

        if disposition == "attachment" or filename:
            attachment_count += 1

    return attachment_count


def count_urls(text):
    """
    Count URLs contained in the email subject and body.
    """

    url_pattern = re.compile(
        r"""
        (?:
            https?://
            |
            www\.
        )
        [^\s<>"']+
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    return len(url_pattern.findall(text))


def create_record(
    subject,
    sender,
    receiver,
    body,
    attachment_count,
    label,
    source,
):
    """
    Create one standard dataset row.
    """

    subject = clean_email_text(subject)
    sender = clean_email_text(sender)
    receiver = clean_email_text(receiver)
    body = clean_email_text(body)

    text_combined = " ".join(
        part
        for part in [
            subject,
            sender,
            body,
        ]
        if part
    ).strip()

    return {
        "subject": subject,
        "sender": sender,
        "receiver": receiver,
        "body": body,
        "text_combined": text_combined,
        "url_count": count_urls(text_combined),
        "attachment_count": attachment_count,
        "message_length": len(text_combined),
        "label": int(label),
        "source": source,
    }


def record_is_usable(record):
    """
    Reject empty, extremely short or corrupted records.
    """

    text = record["text_combined"]

    if not text:
        return False

    if len(text) < 30:
        return False

    if len(text) > 100000:
        return False

    return True

# =========================================================
# Nazario phishing email extraction
# =========================================================

def get_mbox_messages(file_path):
    """
    Read a Nazario mailbox.

    The files do not need to have a .mbox extension.
    """

    messages = []

    try:
        mailbox_file = mailbox.mbox(
            str(file_path),
            factory=lambda raw_message: BytesParser(
                policy=policy.default
            ).parse(raw_message),
        )

        for message in mailbox_file:
            messages.append(message)

    except Exception as error:
        print(
            f"Could not read {file_path.name}: {error}"
        )

    return messages


def load_nazario_emails():
    """
    Extract phishing emails from the Nazario mailbox files.
    """

    records = []

    print("\nLoading Nazario phishing emails...")

    for file_path in NAZARIO_FILES:

        if not file_path.exists():
            print(
                f"Warning: file not found: {file_path}"
            )
            continue

        messages = get_mbox_messages(file_path)

        file_record_count = 0

        for message in messages:

            subject = message.get("subject", "")
            sender = message.get("from", "")
            receiver = message.get("to", "")
            body = extract_email_body(message)

            record = create_record(
                subject=subject,
                sender=sender,
                receiver=receiver,
                body=body,
                attachment_count=count_attachments(message),
                label=1,
                source=f"Nazario-{file_path.name}",
            )

            if not record_is_usable(record):
                continue

            # Skip the mailbox's internal metadata message.
            subject_lower = record["subject"].lower()

            if "don't delete this message" in subject_lower:
                continue

            records.append(record)
            file_record_count += 1

        print(
            f"{file_path.name}: "
            f"{file_record_count:,} phishing emails"
        )

    print(
        f"Total phishing emails collected: "
        f"{len(records):,}"
    )

    return records


# =========================================================
# Dataset cleaning
# =========================================================

def clean_dataset(dataframe):
    """
    Remove unusable rows and duplicates.
    """

    print("\nCleaning combined dataset...")

    original_rows = len(dataframe)

    dataframe = dataframe.copy()

    dataframe["text_combined"] = (
        dataframe["text_combined"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe["text_combined"].str.len() >= 30
    ]

    # Exact duplicate emails.
    dataframe = dataframe.drop_duplicates(
        subset=[
            "text_combined",
            "label",
        ]
    )

    # Remove duplicated text even when metadata differs.
    dataframe = dataframe.drop_duplicates(
        subset=["text_combined"],
        keep="first",
    )

    dataframe = dataframe.reset_index(drop=True)

    removed_rows = original_rows - len(dataframe)

    print(f"Rows removed: {removed_rows:,}")
    print(f"Rows remaining: {len(dataframe):,}")

    return dataframe


def create_balanced_dataset(
    legitimate_dataframe,
    phishing_dataframe,
):
    """
    Balance the final dataset according to the smaller class.
    """

    legitimate_count = len(legitimate_dataframe)
    phishing_count = len(phishing_dataframe)

    target_per_class = min(
        legitimate_count,
        phishing_count,
        MAX_LEGITIMATE_EMAILS,
    )

    if target_per_class == 0:
        raise ValueError(
            "No usable records were collected for one or both classes."
        )

    legitimate_sample = legitimate_dataframe.sample(
        n=target_per_class,
        random_state=RANDOM_STATE,
    )

    phishing_sample = phishing_dataframe.sample(
        n=target_per_class,
        random_state=RANDOM_STATE,
    )

    balanced_dataframe = pd.concat(
        [
            legitimate_sample,
            phishing_sample,
        ],
        ignore_index=True,
    )

    balanced_dataframe = balanced_dataframe.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return balanced_dataframe


# =========================================================
# Main process
# =========================================================

def main():

    print("=" * 70)
    print("BUILDING EMAIL TRAINING DATASET VERSION 2")
    print("=" * 70)

    phishing_records = load_nazario_emails()

    if not phishing_records:
        raise ValueError(
            "No phishing emails were extracted. "
            "Check the phishing-2023, phishing-2024 "
            "and phishing-2025 files."
        )

    phishing_dataframe = pd.DataFrame(
        phishing_records
    )

    phishing_dataframe = clean_dataset(
        phishing_dataframe
    )

    # Collect enough legitimate emails to match phishing.
    legitimate_target = min(
        max(
            len(phishing_dataframe) * 2,
            5000,
        ),
        MAX_LEGITIMATE_EMAILS,
    )

    legitimate_records = load_legitimate_emails(
        limit=legitimate_target
    )

    if not legitimate_records:
        raise ValueError(
            "No legitimate emails were extracted."
        )

    legitimate_dataframe = pd.DataFrame(
        legitimate_records
    )

    legitimate_dataframe = clean_dataset(
        legitimate_dataframe
    )

    final_dataframe = create_balanced_dataset(
        legitimate_dataframe,
        phishing_dataframe,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("EMAIL DATASET VERSION 2 CREATED")
    print("=" * 70)

    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\nFinal dataset shape:")
    print(final_dataframe.shape)

    print("\nFinal label distribution:")
    print(
        final_dataframe["label"]
        .value_counts()
        .sort_index()
    )

    print("\nSource distribution:")
    print(
        final_dataframe["source"]
        .value_counts()
    )

    print("\nColumns:")
    print(final_dataframe.columns.tolist())

    print("\nSample legitimate email:")

    legitimate_example = final_dataframe[
        final_dataframe["label"] == 0
    ].iloc[0]

    print(
        legitimate_example["text_combined"][:500]
    )

    print("\nSample phishing email:")

    phishing_example = final_dataframe[
        final_dataframe["label"] == 1
    ].iloc[0]

    print(
        phishing_example["text_combined"][:500]
    )


if __name__ == "__main__":
    main()
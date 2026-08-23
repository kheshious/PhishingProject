from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "sms_spam.csv"


df = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="latin-1",
)


print("=" * 70)
print("SMS DATASET INSPECTION")
print("=" * 70)


print("\nDataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nFirst 10 records:")
print(
    df.head(10).to_string(
        index=False
    )
)


print("\nMissing values:")
print(
    df.isnull().sum()
)


print("\nLabel distribution:")
print(
    df["label"].value_counts()
)


print("\nLabel percentages:")
print(
    (
        df["label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print("\nDuplicate SMS messages:")
print(
    df.duplicated(
        subset=["text"]
    ).sum()
)


print("\nDuplicate complete records:")
print(
    df.duplicated().sum()
)


print("\nSMS length statistics:")

df["message_length"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.len()
)

print(
    df.groupby("label")[
        "message_length"
    ].describe().round(2)
)


print("\n" + "=" * 70)
print("RANDOM HAM EXAMPLES")
print("=" * 70)

ham_messages = df[
    df["label"] == "ham"
]

for number, text in enumerate(
    ham_messages.sample(
        n=min(15, len(ham_messages)),
        random_state=42,
    )["text"],
    start=1,
):

    print(
        f"\nHAM {number}:"
    )

    print(text)


print("\n" + "=" * 70)
print("RANDOM SPAM EXAMPLES")
print("=" * 70)

spam_messages = df[
    df["label"] == "spam"
]

for number, text in enumerate(
    spam_messages.sample(
        n=min(30, len(spam_messages)),
        random_state=42,
    )["text"],
    start=1,
):

    print(
        f"\nSPAM {number}:"
    )

    print(text)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

SMISHING_DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION"
    / "Dataset_5971"
    / "Dataset_5971.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "sms_training_v2.csv"
)

RANDOM_STATE = 42


df = pd.read_csv(
    SMISHING_DATASET_PATH,
    encoding="utf-8"
)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["text"].str.len() > 0
].copy()

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

df = df[
    df["label"].isin(
        [
            "ham",
            "smishing",
        ]
    )
].copy()

df["target"] = df["label"].map({
    "ham": 0,
    "smishing": 1,
})

ham_df = df[
    df["target"] == 0
].copy()

smishing_df = df[
    df["target"] == 1
].copy()

target_count = min(
    len(ham_df),
    len(smishing_df)
)

ham_sample = ham_df.sample(
    n=target_count,
    random_state=RANDOM_STATE
)

smishing_sample = smishing_df.sample(
    n=target_count,
    random_state=RANDOM_STATE
)

final_df = pd.concat(
    [
        ham_sample,
        smishing_sample,
    ],
    ignore_index=True
)

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

final_df = final_df[
    [
        "text",
        "url",
        "email",
        "phone",
        "target",
    ]
].copy()

final_df = final_df.rename(
    columns={
        "target": "label"
    }
)

final_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("=" * 70)
print("SMS TRAINING DATASET VERSION 2 CREATED")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nDataset shape:")
print(final_df.shape)

print("\nLabel distribution:")
print(
    final_df["label"]
    .value_counts()
    .sort_index()
)

print("\nURL indicator distribution:")
print(
    final_df["url"]
    .value_counts()
)

print("\nEmail indicator distribution:")
print(
    final_df["email"]
    .value_counts()
)

print("\nPhone indicator distribution:")
print(
    final_df["phone"]
    .value_counts()
)

print("\nSample legitimate SMS:")
print(
    final_df[
        final_df["label"] == 0
    ]["text"].iloc[0]
)

print("\nSample smishing SMS:")
print(
    final_df[
        final_df["label"] == 1
    ]["text"].iloc[0]
)
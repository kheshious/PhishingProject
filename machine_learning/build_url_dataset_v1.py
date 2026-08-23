from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "malicious_phish.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "url_training_v1.csv"
)

RANDOM_STATE = 42
MAX_PER_CLASS = 100000


def clean_url(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def main():

    print("=" * 70)
    print("BUILDING URL TRAINING DATASET VERSION 1")
    print("=" * 70)

    print("\nLoading URL dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        usecols=["url", "type"],
        low_memory=False,
    )

    print(f"Rows loaded: {len(df):,}")

    df["url"] = (
        df["url"]
        .apply(clean_url)
    )

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["url"].str.len() > 0
    ].copy()

    print("\nOriginal class distribution:")

    print(
        df["type"]
        .value_counts()
    )

    valid_types = [
        "benign",
        "phishing",
        "defacement",
        "malware",
    ]

    df = df[
        df["type"].isin(valid_types)
    ].copy()

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["url"],
        keep="first",
    )

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print(
        f"\nDuplicate URLs removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Rows after duplicate removal: "
        f"{len(df):,}"
    )

    df["label"] = (
        df["type"]
        .map({
            "benign": 0,
            "phishing": 1,
            "defacement": 1,
            "malware": 1,
        })
        .astype(int)
    )

    df["category"] = df["type"]

    print("\nBinary class distribution before balancing:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    legitimate_df = df[
        df["label"] == 0
    ].copy()

    malicious_df = df[
        df["label"] == 1
    ].copy()

    target_per_class = min(
        len(legitimate_df),
        len(malicious_df),
        MAX_PER_CLASS,
    )

    print(
        f"\nTarget records per class: "
        f"{target_per_class:,}"
    )

    legitimate_sample = legitimate_df.sample(
        n=target_per_class,
        random_state=RANDOM_STATE,
    )

    malicious_sample = malicious_df.sample(
        n=target_per_class,
        random_state=RANDOM_STATE,
    )

    final_df = pd.concat(
        [
            legitimate_sample,
            malicious_sample,
        ],
        ignore_index=True,
    )

    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    final_df = final_df[
        [
            "url",
            "label",
            "category",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("URL TRAINING DATASET VERSION 1 CREATED")
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

    print("\nOriginal malicious category distribution:")

    print(
        final_df["category"]
        .value_counts()
    )

    print("\nSample legitimate URLs:")

    legitimate_examples = (
        final_df[
            final_df["label"] == 0
        ]["url"]
        .head(5)
    )

    for url in legitimate_examples:
        print(url)

    print("\nSample malicious URLs:")

    malicious_examples = (
        final_df[
            final_df["label"] == 1
        ][
            [
                "url",
                "category",
            ]
        ]
        .head(5)
    )

    for _, row in malicious_examples.iterrows():
        print(
            f'{row["category"]}: '
            f'{row["url"]}'
        )


if __name__ == "__main__":
    main()
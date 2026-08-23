from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "malicious_phish.csv"
)


print("=" * 70)
print("URL DATASET INSPECTION")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)

print("\nFile exists:")
print(DATASET_PATH.exists())

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False,
)


print("\nDataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nData types:")
print(df.dtypes)


print("\nFirst 10 records:")
print(
    df.head(10).to_string(index=False)
)


print("\nMissing values:")
print(
    df.isnull().sum()
)


print("\nUnique values per column:")

for column in df.columns:
    print(
        f"{column}: "
        f"{df[column].nunique(dropna=False):,}"
    )


print("\nLabel distribution:")

if "type" in df.columns:

    print(
        df["type"]
        .value_counts(dropna=False)
    )

    print("\nLabel percentages:")

    print(
        (
            df["type"]
            .value_counts(
                normalize=True,
                dropna=False
            )
            * 100
        ).round(2)
    )

else:
    print(
        "No 'type' column found."
    )


print("\nDuplicate complete records:")
print(
    df.duplicated().sum()
)


if "url" in df.columns:

    print("\nDuplicate URLs:")
    print(
        df["url"].duplicated().sum()
    )

    print("\nURL length statistics:")

    url_lengths = (
        df["url"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print(
        url_lengths.describe()
    )


if (
    "url" in df.columns
    and "type" in df.columns
):

    print("\nSample URLs by class:")

    for label in df["type"].dropna().unique():

        print(
            "\n" + "=" * 70
        )

        print(
            f"{str(label).upper()} SAMPLES"
        )

        print(
            "=" * 70
        )

        samples = (
            df[df["type"] == label]["url"]
            .dropna()
            .sample(
                n=min(
                    10,
                    len(
                        df[
                            df["type"] == label
                        ]
                    )
                ),
                random_state=42,
            )
        )

        for number, url in enumerate(
            samples,
            start=1,
        ):
            print(
                f"{number}. {url}"
            )
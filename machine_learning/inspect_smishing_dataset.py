from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION"
    / "Dataset_5971"
    / "Dataset_5971.csv"
)


print("=" * 70)
print("SMISHING DATASET INSPECTION")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)

print("\nFile exists:")
print(DATASET_PATH.exists())

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset was not found: {DATASET_PATH}"
    )


encodings = [
    "utf-8",
    "utf-8-sig",
    "latin-1",
    "windows-1252",
]

df = None
used_encoding = None

for encoding in encodings:
    try:
        df = pd.read_csv(
            DATASET_PATH,
            encoding=encoding,
        )

        used_encoding = encoding
        break

    except UnicodeDecodeError:
        continue


if df is None:
    raise ValueError(
        "The dataset could not be read using the supported encodings."
    )


print("\nEncoding used:")
print(used_encoding)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst 10 records:")
print(
    df.head(10).to_string(
        index=False
    )
)

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values per column:")

for column in df.columns:
    print(
        f"{column}: "
        f"{df[column].nunique(dropna=False):,}"
    )


print("\nPossible label values:")

for column in df.columns:

    unique_count = df[column].nunique(
        dropna=True
    )

    if unique_count <= 20:

        print(f"\n{column}:")

        print(
            df[column]
            .value_counts(dropna=False)
        )


print("\nDuplicate complete records:")
print(
    df.duplicated().sum()
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
import pandas as pd

DATASET_PATH = "datasets/phishing_email.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\n" + "=" * 80)
print("20 SAMPLE LABEL 0 EMAILS")
print("=" * 80)

for index, text in enumerate(
    df[df["label"] == 0]["text_combined"]
    .sample(20, random_state=42),
    start=1
):
    print(f"\nLABEL 0 - SAMPLE {index}")
    print(text[:500])

print("\n" + "=" * 80)
print("20 SAMPLE LABEL 1 EMAILS")
print("=" * 80)

for index, text in enumerate(
    df[df["label"] == 1]["text_combined"]
    .sample(20, random_state=42),
    start=1
):
    print(f"\nLABEL 1 - SAMPLE {index}")
    print(text[:500])
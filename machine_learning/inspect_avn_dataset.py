import pandas as pd

df = pd.read_csv("datasets/AVN_Corpus.csv")

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\n====================")
print("LABEL 0")
print("====================")

for i, row in df[df["label"] == 0].sample(10, random_state=42).iterrows():
    print("\nSUBJECT:", row["subject"])
    print("BODY:")
    print(str(row["body"])[:500])

print("\n====================")
print("LABEL 1")
print("====================")

for i, row in df[df["label"] == 1].sample(10, random_state=42).iterrows():
    print("\nSUBJECT:", row["subject"])
    print("BODY:")
    print(str(row["body"])[:500])
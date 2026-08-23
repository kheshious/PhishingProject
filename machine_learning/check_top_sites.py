import pandas as pd

df = pd.read_csv(
    "datasets/top-1m.csv",
    header=None,
    names=["rank", "domain"]
)

print(df.head(20))

print("\nRows:", len(df))

print("\nColumns:")
print(df.columns)
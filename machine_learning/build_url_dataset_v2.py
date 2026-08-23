from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import html

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MALICIOUS_DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "malicious_phish.csv"
)

POPULAR_DOMAINS_PATH = (
    BASE_DIR
    / "datasets"
    / "top-1m.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "url_training_v2.csv"
)

RANDOM_STATE = 42

BENIGN_FROM_ORIGINAL = 50000
BENIGN_FROM_POPULAR = 50000
MALICIOUS_TARGET = 100000


def normalise_url(value):

    if pd.isna(value):
        return ""

    value = html.unescape(
        str(value).strip()
    )

    if not value:
        return ""

    parsing_value = value

    if "://" not in parsing_value:
        parsing_value = (
            "http://"
            + parsing_value
        )

    try:
        parsed = urlsplit(
            parsing_value
        )
    except ValueError:
        return ""

    hostname = (
        parsed.hostname or ""
    ).lower().strip(".")

    if not hostname:
        return ""

    if hostname.startswith("www."):
        hostname = hostname[4:]

    try:
        port = parsed.port
    except ValueError:
        port = None

    if port is not None:
        hostname = (
            f"{hostname}:{port}"
        )

    path = parsed.path or ""

    if path == "/":
        path = ""

    query = parsed.query or ""

    normalised = urlunsplit(
        (
            "",
            hostname,
            path,
            query,
            "",
        )
    )

    normalised = normalised.lstrip("//")

    return normalised.strip()


def main():

    print("=" * 70)
    print("BUILDING URL TRAINING DATASET VERSION 2")
    print("=" * 70)

    print("\nLoading malicious URL dataset...")

    url_df = pd.read_csv(
        MALICIOUS_DATASET_PATH,
        usecols=[
            "url",
            "type",
        ],
        low_memory=False,
    )

    print(
        f"Rows loaded: "
        f"{len(url_df):,}"
    )

    url_df["type"] = (
        url_df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    url_df["normalised_url"] = (
        url_df["url"]
        .apply(normalise_url)
    )

    url_df = url_df[
        url_df["normalised_url"]
        .str.len() > 0
    ].copy()

    url_df = url_df[
        url_df["type"].isin(
            [
                "benign",
                "phishing",
                "defacement",
                "malware",
            ]
        )
    ].copy()

    before_duplicates = len(
        url_df
    )

    url_df = url_df.drop_duplicates(
        subset=[
            "normalised_url"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "\nDuplicates removed from "
        "malicious_phish.csv:",
        f"{before_duplicates - len(url_df):,}"
    )

    print(
        "\nOriginal class distribution "
        "after normalisation:"
    )

    print(
        url_df["type"]
        .value_counts()
    )

    print(
        "\nLoading popular legitimate domains..."
    )

    popular_df = pd.read_csv(
        POPULAR_DOMAINS_PATH,
        header=None,
        names=[
            "rank",
            "domain",
        ],
        usecols=[
            0,
            1,
        ],
    )

    print(
        f"Popular domains loaded: "
        f"{len(popular_df):,}"
    )

    popular_df["normalised_url"] = (
        popular_df["domain"]
        .apply(normalise_url)
    )

    popular_df = popular_df[
        popular_df[
            "normalised_url"
        ].str.len() > 0
    ].copy()

    popular_df = (
        popular_df
        .drop_duplicates(
            subset=[
                "normalised_url"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    benign_original = (
        url_df[
            url_df["type"]
            == "benign"
        ]
        .copy()
    )

    malicious_df = (
        url_df[
            url_df["type"].isin(
                [
                    "phishing",
                    "defacement",
                    "malware",
                ]
            )
        ]
        .copy()
    )

    original_sample_size = min(
        BENIGN_FROM_ORIGINAL,
        len(
            benign_original
        ),
    )

    popular_sample_size = min(
        BENIGN_FROM_POPULAR,
        len(
            popular_df
        ),
    )

    malicious_sample_size = min(
        MALICIOUS_TARGET,
        len(
            malicious_df
        ),
    )

    benign_original_sample = (
        benign_original.sample(
            n=original_sample_size,
            random_state=RANDOM_STATE,
        )[
            [
                "normalised_url",
            ]
        ]
        .copy()
    )

    benign_original_sample[
        "label"
    ] = 0

    benign_original_sample[
        "category"
    ] = "benign"

    benign_original_sample[
        "source"
    ] = "malicious_phish"

    popular_sample = (
        popular_df
        .sort_values("rank")
        .head(
            popular_sample_size
        )[
            [
                "normalised_url",
            ]
        ]
        .copy()
    )

    popular_sample[
        "label"
    ] = 0

    popular_sample[
        "category"
    ] = "benign"

    popular_sample[
        "source"
    ] = "top_1m"

    malicious_sample = (
        malicious_df.sample(
            n=malicious_sample_size,
            random_state=RANDOM_STATE,
        )[
            [
                "normalised_url",
                "type",
            ]
        ]
        .copy()
    )

    malicious_sample = (
        malicious_sample
        .rename(
            columns={
                "type":
                "category"
            }
        )
    )

    malicious_sample[
        "label"
    ] = 1

    malicious_sample[
        "source"
    ] = "malicious_phish"

    final_df = pd.concat(
        [
            benign_original_sample,
            popular_sample,
            malicious_sample,
        ],
        ignore_index=True,
    )

    before_final_duplicates = (
        len(final_df)
    )

    final_df = (
        final_df
        .drop_duplicates(
            subset=[
                "normalised_url"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nDuplicates removed after "
        "combining sources:",
        (
            before_final_duplicates
            - len(final_df)
        ),
    )

    legitimate_df = (
        final_df[
            final_df["label"]
            == 0
        ]
        .copy()
    )

    malicious_final_df = (
        final_df[
            final_df["label"]
            == 1
        ]
        .copy()
    )

    final_target = min(
        len(legitimate_df),
        len(malicious_final_df),
    )

    legitimate_df = (
        legitimate_df.sample(
            n=final_target,
            random_state=RANDOM_STATE,
        )
    )

    malicious_final_df = (
        malicious_final_df.sample(
            n=final_target,
            random_state=RANDOM_STATE,
        )
    )

    final_df = pd.concat(
        [
            legitimate_df,
            malicious_final_df,
        ],
        ignore_index=True,
    )

    final_df = (
        final_df.sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(
            drop=True
        )
    )

    final_df = final_df[
        [
            "normalised_url",
            "label",
            "category",
            "source",
        ]
    ]

    final_df = final_df.rename(
        columns={
            "normalised_url":
            "url"
        }
    )

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
    print("URL TRAINING DATASET VERSION 2 CREATED")
    print("=" * 70)

    print("\nSaved to:")
    print(
        OUTPUT_PATH
    )

    print(
        "\nFinal dataset shape:"
    )

    print(
        final_df.shape
    )

    print(
        "\nLabel distribution:"
    )

    print(
        final_df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSource distribution:"
    )

    print(
        final_df["source"]
        .value_counts()
    )

    print(
        "\nCategory distribution:"
    )

    print(
        final_df["category"]
        .value_counts()
    )

    print(
        "\nSample legitimate URLs:"
    )

    for url in (
        final_df[
            final_df["label"]
            == 0
        ]["url"]
        .head(10)
    ):
        print(
            url
        )

    print(
        "\nSample malicious URLs:"
    )

    sample_malicious = (
        final_df[
            final_df["label"]
            == 1
        ][
            [
                "url",
                "category",
            ]
        ]
        .head(10)
    )

    for _, row in (
        sample_malicious
        .iterrows()
    ):
        print(
            f'{row["category"]}: '
            f'{row["url"]}'
        )


if __name__ == "__main__":
    main()
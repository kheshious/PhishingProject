from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import html
import ipaddress
import re

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "best_url_model_v2.joblib"
VECTORIZER_PATH = MODEL_DIR / "url_vectorizer_v2.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def normalise_url(value):

    if not isinstance(value, str):
        return ""

    value = html.unescape(
        value.strip()
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


def get_hostname(url):

    normalised = normalise_url(
        url
    )

    if not normalised:
        return ""

    parsing_value = (
        "http://"
        + normalised
    )

    try:
        parsed = urlsplit(
            parsing_value
        )
    except ValueError:
        return ""

    return (
        parsed.hostname or ""
    ).lower()


def validate_url(url):

    if not isinstance(url, str):
        return False, (
            "Please enter a website URL."
        )

    cleaned = url.strip()

    if not cleaned:
        return False, (
            "Please enter a website URL."
        )

    if any(
        character.isspace()
        for character in cleaned
    ):
        return False, (
            "The submitted value contains spaces "
            "and is not a valid website URL."
        )

    normalised = normalise_url(
        cleaned
    )

    if not normalised:
        return False, (
            "The submitted value is not a valid website URL."
        )

    hostname = get_hostname(
        cleaned
    )

    if not hostname:
        return False, (
            "The submitted value does not contain "
            "a valid website domain."
        )

    try:
        ipaddress.ip_address(
            hostname
        )

        return True, None

    except ValueError:
        pass

    if "." not in hostname:
        return False, (
            "Please enter a complete website URL "
            "with a valid domain extension."
        )

    if len(hostname) > 253:
        return False, (
            "The website domain is too long."
        )

    labels = hostname.split(".")

    domain_pattern = re.compile(
        r"^[a-zA-Z0-9-]+$"
    )

    for label in labels:

        if not label:
            return False, (
                "The website domain has an invalid format."
            )

        if not domain_pattern.fullmatch(
            label
        ):
            return False, (
                "The website domain contains "
                "invalid characters."
            )

        if (
            label.startswith("-")
            or label.endswith("-")
        ):
            return False, (
                "The website domain contains "
                "an invalid hyphen position."
            )

        if len(label) > 63:
            return False, (
                "Part of the website domain is too long."
            )

    top_level_domain = labels[-1]

    if (
        len(top_level_domain) < 2
        or not top_level_domain.isalpha()
    ):
        return False, (
            "The website domain extension "
            "does not appear to be valid."
        )

    return True, None


def get_warnings(url):

    warnings = []

    if url.lower().startswith(
        "http://"
    ):
        warnings.append(
            "This website uses HTTP instead of HTTPS. "
            "Information sent to the website may not be encrypted."
        )

    return warnings


def predict_url(url):

    is_valid, validation_error = (
        validate_url(
            url
        )
    )

    if not is_valid:

        return {
            "is_valid": False,
            "validation_error": validation_error,
            "prediction": "Invalid URL",
            "risk_level": "None",
            "confidence": 0,
            "legitimate_probability": 0,
            "malicious_probability": 0,
            "normalised_url": "",
            "hostname": "",
            "warnings": [],
        }

    normalised = normalise_url(
        url
    )

    vectorised_url = (
        vectorizer.transform(
            [normalised]
        )
    )

    predicted_class = int(
        model.predict(
            vectorised_url
        )[0]
    )

    probabilities = (
        model.predict_proba(
            vectorised_url
        )[0]
    )

    classes = list(
        model.classes_
    )

    legitimate_index = (
        classes.index(0)
    )

    malicious_index = (
        classes.index(1)
    )

    legitimate_probability = float(
        probabilities[
            legitimate_index
        ] * 100
    )

    malicious_probability = float(
        probabilities[
            malicious_index
        ] * 100
    )

    predicted_index = (
        classes.index(
            predicted_class
        )
    )

    confidence = float(
        probabilities[
            predicted_index
        ] * 100
    )

    if predicted_class == 1:

        prediction = "Malicious"
        risk_level = "High"

    else:

        prediction = "Legitimate"
        risk_level = "Low"

    return {
        "is_valid": True,
        "validation_error": None,
        "prediction": prediction,
        "risk_level": risk_level,
        "confidence": round(
            confidence,
            2
        ),
        "legitimate_probability": round(
            legitimate_probability,
            2
        ),
        "malicious_probability": round(
            malicious_probability,
            2
        ),
        "normalised_url": normalised,
        "hostname": get_hostname(
            url
        ),
        "warnings": get_warnings(
            url
        ),
    }


if __name__ == "__main__":

    test_urls = [
        "google.com",
        "https://www.google.com",
        "http://google.com",
        "github.com",
        "https://github.com",
        "microsoft.com",
        "https://www.microsoft.com",
        "youtube.com",
        "https://www.youtube.com",
        "https://secure-account-login-example.com",
        "http://192.168.1.10/login",
        "https://account-verification-example.com/login",
        "hello",
        "hsgstry",
    ]

    for url in test_urls:

        result = predict_url(
            url
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "Submitted URL:",
            url
        )

        print(
            "Valid:",
            result["is_valid"]
        )

        if not result["is_valid"]:

            print(
                "Prediction:",
                result["prediction"]
            )

            print(
                "Validation error:",
                result[
                    "validation_error"
                ]
            )

            continue

        print(
            "Normalised URL:",
            result[
                "normalised_url"
            ]
        )

        print(
            "Prediction:",
            result["prediction"]
        )

        print(
            "Risk:",
            result["risk_level"]
        )

        print(
            "Confidence:",
            f'{result["confidence"]}%'
        )

        print(
            "Legitimate probability:",
            f'{result["legitimate_probability"]}%'
        )

        print(
            "Malicious probability:",
            f'{result["malicious_probability"]}%'
        )

        if result["warnings"]:

            print(
                "Warnings:"
            )

            for warning in result[
                "warnings"
            ]:

                print(
                    "-",
                    warning
                )
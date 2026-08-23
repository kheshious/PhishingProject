from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import html
import ipaddress
import re

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from url_features import extract_feature_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "best_url_model_v3.joblib"
VECTORIZER_PATH = MODEL_DIR / "url_vectorizer_v3.joblib"
SCALER_PATH = MODEL_DIR / "url_feature_scaler_v3.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
feature_scaler = joblib.load(SCALER_PATH)


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
        parsing_value = "http://" + parsing_value

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

    return normalised.lstrip("//").strip()


def get_hostname(url):

    normalised = normalise_url(
        url
    )

    if not normalised:
        return ""

    try:
        parsed = urlsplit(
            "http://" + normalised
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

    cleaned_url = url.strip()

    if not cleaned_url:
        return False, (
            "Please enter a website URL."
        )

    if any(
        character.isspace()
        for character in cleaned_url
    ):
        return False, (
            "The submitted value contains spaces "
            "and is not a valid website URL."
        )

    normalised = normalise_url(
        cleaned_url
    )

    if not normalised:
        return False, (
            "The submitted value is not a valid website URL."
        )

    hostname = get_hostname(
        cleaned_url
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

    domain_pattern = re.compile(
        r"^[a-zA-Z0-9-]+$"
    )

    labels = hostname.split(".")

    for label in labels:

        if not label:
            return False, (
                "The website domain has an invalid format."
            )

        if not domain_pattern.fullmatch(
            label
        ):
            return False, (
                "The website domain contains invalid characters."
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


def prepare_model_input(url):

    normalised = normalise_url(
        url
    )

    text_features = vectorizer.transform(
        [normalised]
    )

    lexical_features = extract_feature_matrix(
        [normalised]
    )

    lexical_features = np.asarray(
        lexical_features,
        dtype=float,
    )

    lexical_features_scaled = (
        feature_scaler.transform(
            lexical_features
        )
    )

    combined_features = hstack(
        [
            text_features,
            csr_matrix(
                lexical_features_scaled
            ),
        ],
        format="csr",
    )

    return (
        combined_features,
        normalised,
    )


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


def build_explanation(
    predicted_class,
    confidence,
):

    if predicted_class == 1:

        reasons = [
            (
                "The trained machine-learning model detected "
                "URL patterns associated with malicious websites."
            )
        ]

        recommended_actions = [
            "Do not enter passwords, banking details or personal information.",
            "Avoid downloading files from the website.",
            "Do not continue using the link unless you can verify it.",
            "Report the URL if you believe it is fraudulent or harmful.",
        ]

    else:

        reasons = [
            (
                "The trained machine-learning model did not detect "
                "strong patterns associated with malicious URLs."
            )
        ]

        recommended_actions = [
            (
                "No immediate action is required, but verify "
                "unexpected websites before entering sensitive information."
            )
        ]

    return (
        reasons,
        recommended_actions,
    )


def analyse_url_segments(
    url,
    predicted_class,
):

    segment_type = (
        "suspicious"
        if predicted_class == 1
        else "normal"
    )

    return [
        {
            "text": url,
            "type": segment_type,
            "label": "",
            "reason": (
                "The complete URL was analysed "
                "by the machine-learning model."
            ),
        }
    ]


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
            "decision_source": "Input Validation",
            "model_prediction": "N/A",
            "model_display_prediction": "N/A",
            "model_confidence": 0,
            "confidence": 0,
            "legitimate_probability": 0,
            "malicious_probability": 0,
            "class_probabilities": {},
            "reasons": [
                validation_error
            ],
            "warnings": [],
            "recommended_actions": [
                "Enter a complete website URL.",
                "Example: https://www.google.com",
            ],
            "url_segments": [],
            "hostname": "",
            "normalised_url": "",
            "detected_brands": [],
            "impersonated_brands": [],
        }

    model_input, normalised = (
        prepare_model_input(
            url
        )
    )

    predicted_class = int(
        model.predict(
            model_input
        )[0]
    )

    probabilities = model.predict_proba(
        model_input
    )[0]

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

    reasons, recommended_actions = (
        build_explanation(
            predicted_class,
            confidence,
        )
    )

    warnings = get_warnings(
        url
    )

    url_segments = analyse_url_segments(
        url,
        predicted_class,
    )

    return {
        "is_valid": True,
        "validation_error": None,

        "prediction": prediction,
        "risk_level": risk_level,
        "decision_source": (
            "Machine Learning Model"
        ),

        "model_prediction": predicted_class,
        "model_display_prediction": prediction,

        "model_confidence": round(
            confidence,
            2
        ),

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

        "class_probabilities": {
            "legitimate": round(
                legitimate_probability,
                2
            ),
            "malicious": round(
                malicious_probability,
                2
            ),
        },

        "reasons": reasons,
        "warnings": warnings,

        "recommended_actions": (
            recommended_actions
        ),

        "url_segments": url_segments,

        "hostname": get_hostname(
            url
        ),

        "normalised_url": normalised,

        "detected_brands": [],
        "impersonated_brands": [],
    }


if __name__ == "__main__":

    test_urls = [
        "google.com",
        "https://www.google.com",
        "github.com",
        "https://github.com",
        "microsoft.com",
        "https://www.microsoft.com",
        "youtube.com",
        "amazon.com",
        "apple.com",
        "linkedin.com",
        "https://secure-account-login-example.com",
        "https://account-verification-example.com/login",
        "http://192.168.1.10/login",
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

        print(
            "Prediction:",
            result["prediction"]
        )

        if not result["is_valid"]:

            print(
                "Validation error:",
                result["validation_error"]
            )

            continue

        print(
            "Normalised URL:",
            result["normalised_url"]
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
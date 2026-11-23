
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from machine_learning.predict_email import model, vectorizer, predict_email


def inspect_model():
    features = vectorizer.get_feature_names_out()

    phishing_index = list(model.classes_).index(1)
    legitimate_index = list(model.classes_).index(0)

    phishing_log_prob = model.feature_log_prob_[phishing_index]
    legitimate_log_prob = model.feature_log_prob_[legitimate_index]

    # Positive values indicate features more associated
    # with phishing than legitimate messages.
    differences = phishing_log_prob - legitimate_log_prob

    top_phishing = np.argsort(differences)[-20:][::-1]
    top_legitimate = np.argsort(differences)[:20]

    print("\nTOP 20 PHISHING-ASSOCIATED FEATURES")
    print("-" * 55)

    for index in top_phishing:
        print(
            f"{features[index]:30} "
            f"{differences[index]:.4f}"
        )

    print("\nTOP 20 LEGITIMATE-ASSOCIATED FEATURES")
    print("-" * 55)

    for index in top_legitimate:
        print(
            f"{features[index]:30} "
            f"{differences[index]:.4f}"
        )


def inspect_text(text):
    transformed = vectorizer.transform([text])

    feature_names = vectorizer.get_feature_names_out()

    phishing_index = list(model.classes_).index(1)
    legitimate_index = list(model.classes_).index(0)

    differences = (
        model.feature_log_prob_[phishing_index]
        - model.feature_log_prob_[legitimate_index]
    )

    print("\nFEATURES FOUND IN TEST MESSAGE")
    print("-" * 55)

    contributions = []

    for index, value in zip(
        transformed.indices,
        transformed.data
    ):
        contribution = value * differences[index]

        contributions.append((
            feature_names[index],
            contribution
        ))

    contributions.sort(
        key=lambda item: abs(item[1]),
        reverse=True
    )

    for feature, contribution in contributions[:25]:
        direction = (
            "Phishing"
            if contribution > 0
            else "Legitimate"
        )

        print(
            f"{feature:30} "
            f"{contribution:+.4f} "
            f"{direction}"
        )

    result = predict_email(text)

    print("\nMODEL PREDICTION")
    print("Prediction:", result["prediction"])
    print(
        "Phishing probability:",
        result["phishing_probability"],
        "%"
    )


if __name__ == "__main__":
    inspect_model()

    test_email = """
    Hello,

    I am sending this message to confirm that
    everything is working correctly.

    No action is required.

    Thank you.
    """

    inspect_text(test_email)

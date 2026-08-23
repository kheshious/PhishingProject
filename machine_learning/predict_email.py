from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "best_email_model_v2.joblib"
VECTORIZER_PATH = MODEL_DIR / "email_vectorizer_v2.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_email(email_text: str) -> dict:

    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("Email content is required.")

    email_text = email_text.strip()

    vectorised_text = vectorizer.transform(
        [email_text]
    )

    predicted_class = int(
        model.predict(vectorised_text)[0]
    )

    probabilities = model.predict_proba(
        vectorised_text
    )[0]

    class_index = list(
        model.classes_
    ).index(predicted_class)

    confidence = float(
        probabilities[class_index] * 100
    )

    phishing_index = list(
        model.classes_
    ).index(1)

    phishing_probability = float(
        probabilities[phishing_index] * 100
    )

    legitimate_index = list(
        model.classes_
    ).index(0)

    legitimate_probability = float(
        probabilities[legitimate_index] * 100
    )

    if predicted_class == 1:

        prediction = "Phishing"
        risk_level = "High"

        reasons = [
            (
                "The trained machine-learning model detected "
                "patterns associated with phishing emails."
            )
        ]

        recommended_actions = [
            "Avoid clicking links or opening unexpected attachments.",
            "Do not provide passwords, PINs, OTPs or banking details.",
            "Verify the sender through an official communication channel.",
            "Report the email if you believe it is fraudulent.",
        ]

    else:

        prediction = "Legitimate"
        risk_level = "Low"

        reasons = [
            (
                "The trained machine-learning model did not detect "
                "strong patterns associated with phishing."
            )
        ]

        recommended_actions = [
            (
                "No immediate action is required, but always verify "
                "unexpected requests before sharing sensitive information."
            )
        ]

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "risk_level": risk_level,
        "model_prediction": predicted_class,
        "phishing_probability": round(
            phishing_probability,
            2
        ),
        "legitimate_probability": round(
            legitimate_probability,
            2
        ),
        "reasons": reasons,
        "recommended_actions": recommended_actions,
    }


if __name__ == "__main__":

    test_emails = [
        """
        Hi Sarah,

        This is a reminder that our project review meeting
        is scheduled for tomorrow at 10:00 AM.

        Please bring your updated presentation slides.

        Kind regards,
        John
        """,

        """
        Dear Customer,

        Your monthly bank statement is now available.

        Please access your statement using the official
        banking application or website.

        Thank you for banking with us.
        """,

        """
        Subject: Urgent Account Verification

        Your account has been suspended.

        Click the link below immediately to verify
        your identity.

        http://secure-account-login-example.com

        Failure to verify within 24 hours will result
        in permanent account suspension.
        """,

        """
        Subject: Mailbox Security Alert

        Your mailbox will be disabled.

        Sign in immediately using the link below to
        confirm your account credentials.

        http://mail-security-verification-example.com
        """,
    ]

    for email in test_emails:

        result = predict_email(email)

        print("\n" + "=" * 70)

        print(
            email.strip()[:200]
        )

        print(
            "\nPrediction:",
            result["prediction"]
        )

        print(
            "Confidence:",
            f'{result["confidence"]}%'
        )

        print(
            "Risk:",
            result["risk_level"]
        )

        print(
            "Legitimate probability:",
            f'{result["legitimate_probability"]}%'
        )

        print(
            "Phishing probability:",
            f'{result["phishing_probability"]}%'
        )
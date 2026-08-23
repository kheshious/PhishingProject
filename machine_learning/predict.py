from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "best_sms_model_v2.joblib"
VECTORIZER_PATH = MODEL_DIR / "sms_vectorizer_v2.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_message(message: str) -> dict:

    if not isinstance(message, str) or not message.strip():
        raise ValueError("SMS content is required.")

    message = message.strip()

    vectorised_message = vectorizer.transform(
        [message]
    )

    predicted_class = int(
        model.predict(vectorised_message)[0]
    )

    probabilities = model.predict_proba(
        vectorised_message
    )[0]

    class_index = list(
        model.classes_
    ).index(predicted_class)

    confidence = float(
        probabilities[class_index] * 100
    )

    legitimate_index = list(
        model.classes_
    ).index(0)

    smishing_index = list(
        model.classes_
    ).index(1)

    legitimate_probability = float(
        probabilities[legitimate_index] * 100
    )

    smishing_probability = float(
        probabilities[smishing_index] * 100
    )

    if predicted_class == 1:

        prediction = "Phishing"
        risk_level = "High"

        reasons = [
            (
                "The trained machine-learning model detected "
                "patterns associated with smishing messages."
            )
        ]

        recommended_actions = [
            "Do not reply to the message.",
            "Do not click unexpected links.",
            "Do not provide passwords, PINs, OTPs or banking details.",
            "Verify the message using an official communication channel.",
            "Report the message if you believe it is fraudulent.",
        ]

    else:

        prediction = "Legitimate"
        risk_level = "Low"

        reasons = [
            (
                "The trained machine-learning model did not detect "
                "strong patterns associated with smishing."
            )
        ]

        recommended_actions = [
            (
                "No immediate action is required, but verify unexpected "
                "requests before sharing sensitive information."
            )
        ]

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "risk_level": risk_level,
        "model_prediction": predicted_class,
        "legitimate_probability": round(
            legitimate_probability,
            2
        ),
        "smishing_probability": round(
            smishing_probability,
            2
        ),
        "reasons": reasons,
        "recommended_actions": recommended_actions,
    }


if __name__ == "__main__":

    test_messages = [
        """
        Hi Sarah, are we still meeting tomorrow at 10?
        Let me know if anything changes.
        """,

        """
        Your appointment is confirmed for Monday at 09:00.
        Please arrive 15 minutes early.
        """,

        """
        URGENT: Your bank account has been suspended.
        Verify your details immediately at
        http://secure-bank-check-example.com
        """,

        """
        Congratulations! You have been selected to receive
        a R5000 cash prize. Click the link below to claim
        your reward now.
        http://claim-prize-example.com
        """,

        """
        Your parcel could not be delivered.
        Pay the outstanding delivery fee using the link below:
        http://delivery-payment-example.com
        """,
    ]

    for message in test_messages:

        result = predict_message(message)

        print("\n" + "=" * 70)

        print(message.strip())

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
            "Smishing probability:",
            f'{result["smishing_probability"]}%'
        )
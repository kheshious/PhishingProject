import re


def generate_explanation(message: str, result: str, phishing_probability: float):
    reasons = []
    recommended_actions = []

    lower_message = message.lower()

    if phishing_probability < 40:
        return (
            ["No suspicious phishing indicators were detected. The message appears to be legitimate."],
            ["No action required."]
        )

    # Check for suspicious link
    if re.search(r"http\S+|www\S+", lower_message):
        reasons.append("A web link was found in the message.")
        recommended_actions.append("Do not click unknown or unexpected links.")

    # Check urgent language
    urgent_words = ["urgent", "immediately", "now", "suspended", "blocked", "verify"]
    if any(word in lower_message for word in urgent_words):
        reasons.append("The message uses urgent language to pressure the user.")
        recommended_actions.append("Do not rush. Verify the message before taking action.")

    # Check prize/reward wording
    prize_words = ["won", "winner", "prize", "claim", "free", "cash", "reward"]
    if any(word in lower_message for word in prize_words):
        reasons.append("The message mentions a prize, reward, or free offer.")
        recommended_actions.append("Be careful of unexpected prize or reward messages.")

    # Check banking/account wording
    banking_words = ["bank", "account", "password", "pin", "otp", "login"]
    if any(word in lower_message for word in banking_words):
        reasons.append("The message refers to banking, login, or account information.")
        recommended_actions.append("Never share passwords, PINs, OTPs, or banking details.")

    # If probability is medium/high but no simple rule matched
    if not reasons:
        reasons.append("The machine learning model detected patterns commonly linked to phishing.")
        recommended_actions.append("Review the message carefully before clicking links or sharing information.")

    # Stronger recommendation for high-risk cases
    if phishing_probability >= 70:
        recommended_actions.append("Report this message as a scam and avoid interacting with it.")

    return reasons, list(set(recommended_actions))
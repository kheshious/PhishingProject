import pandas as pd
import joblib

from preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


dataset_path = "datasets/sms_spam.csv"

df = pd.read_csv(dataset_path, sep="\t", names=["label", "message"])

df["clean_message"] = df["message"].apply(clean_text)

df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

X = df["clean_message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

best_model_name = None
best_f1 = 0
best_model = None

for name, model in models.items():
    print("\n==============================")
    print(f"Training {name}")
    print("==============================")

    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print("\nBest model:", best_model_name)
print("Best F1-score:", round(best_f1, 4))

joblib.dump(best_model, "model/best_sms_model.joblib")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")

print("\nModel and vectorizer saved successfully.")
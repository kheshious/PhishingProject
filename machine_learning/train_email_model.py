import joblib
import pandas as pd

from preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Loading phishing email dataset...")

df = pd.read_csv("datasets/phishing_email.csv")

df["clean_text"] = df["text_combined"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

best_model = None
best_name = ""
best_f1 = 0

for name, model in models.items():

    print("\n===========================")
    print(name)
    print("===========================")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Accuracy :", round(accuracy,4))
    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1-score :", round(f1,4))

    print(classification_report(y_test,predictions))

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

joblib.dump(best_model,"model/best_email_model.joblib")
joblib.dump(vectorizer,"model/email_vectorizer.joblib")

print("\n===========================")
print("Best Model:",best_name)
print("Best F1:",round(best_f1,4))
print("===========================")

print("\nEmail model saved successfully.")
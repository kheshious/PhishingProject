from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "sms_training_v2.csv"
)

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "best_sms_model_v2.joblib"
)

VECTORIZER_PATH = (
    MODEL_DIR
    / "sms_vectorizer_v2.joblib"
)

RESULTS_PATH = (
    MODEL_DIR
    / "sms_model_results_v2.csv"
)


print("=" * 70)
print("SMS MODEL VERSION 2 TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Rows loaded:", len(df))

print("\nClass distribution:")
print(
    df["label"]
    .value_counts()
    .sort_index()
)

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["text"].str.len() > 0
].copy()

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)


X = df["text"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Validation records:",
    len(X_validation)
)

print(
    "Testing records   :",
    len(X_test)
)


print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    sublinear_tf=True,
)

X_train_vectorised = vectorizer.fit_transform(
    X_train
)

X_validation_vectorised = vectorizer.transform(
    X_validation
)

X_test_vectorised = vectorizer.transform(
    X_test
)

print(
    "TF-IDF features:",
    X_train_vectorised.shape[1]
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}


results = []

best_model = None
best_model_name = None
best_validation_f1 = -1

for model_name, current_model in models.items():

    print("\n" + "=" * 70)
    print(f"Training {model_name}")
    print("=" * 70)

    start_time = time.time()

    current_model.fit(
        X_train_vectorised,
        y_train
    )

    training_time = (
        time.time() - start_time
    )

    validation_predictions = current_model.predict(
        X_validation_vectorised
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    validation_recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    print(
        f"Validation Accuracy  : "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Validation Precision : "
        f"{validation_precision:.4f}"
    )

    print(
        f"Validation Recall    : "
        f"{validation_recall:.4f}"
    )

    print(
        f"Validation F1-score  : "
        f"{validation_f1:.4f}"
    )

    print(
        f"Training time        : "
        f"{training_time:.2f} seconds"
    )

    results.append({
        "Model": model_name,
        "Validation Accuracy": round(
            validation_accuracy,
            4
        ),
        "Validation Precision": round(
            validation_precision,
            4
        ),
        "Validation Recall": round(
            validation_recall,
            4
        ),
        "Validation F1": round(
            validation_f1,
            4
        ),
        "Training Time Seconds": round(
            training_time,
            2
        ),
    })

    if validation_f1 > best_validation_f1:
        best_validation_f1 = validation_f1
        best_model = current_model
        best_model_name = model_name


results_dataframe = pd.DataFrame(
    results
)

results_dataframe = (
    results_dataframe
    .sort_values(
        by="Validation F1",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("SMS MODEL VERSION 2 VALIDATION COMPARISON")
print("=" * 70)

print(
    results_dataframe.to_string(
        index=False
    )
)

print("\n" + "=" * 70)

print(
    f"Best Model: {best_model_name}"
)

print(
    f"Best Validation F1-score: "
    f"{best_validation_f1:.4f}"
)

print("=" * 70)


print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

test_predictions = best_model.predict(
    X_test_vectorised
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1-score  : {test_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Legitimate",
            "Smishing"
        ],
        zero_division=0,
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


joblib.dump(
    best_model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

results_dataframe.to_csv(
    RESULTS_PATH,
    index=False
)


print("\nSaved successfully:")

print(
    f"- {MODEL_PATH}"
)

print(
    f"- {VECTORIZER_PATH}"
)

print(
    f"- {RESULTS_PATH}"
)
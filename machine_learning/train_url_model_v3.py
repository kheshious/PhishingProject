from pathlib import Path
import time

import joblib
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
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

from url_features import extract_feature_matrix


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "url_training_v2.csv"
)

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / "best_url_model_v3.joblib"
)

VECTORIZER_PATH = (
    MODEL_DIR
    / "url_vectorizer_v3.joblib"
)

SCALER_PATH = (
    MODEL_DIR
    / "url_feature_scaler_v3.joblib"
)

RESULTS_PATH = (
    MODEL_DIR
    / "url_model_results_v3.csv"
)


print("=" * 70)
print("URL MODEL VERSION 3 TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False,
)

print(
    f"Rows loaded: {len(df):,}"
)

df["url"] = (
    df["url"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["url"].str.len() > 0
].copy()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce",
)

df = df.dropna(
    subset=["label"]
).copy()

df["label"] = (
    df["label"]
    .astype(int)
)

df = df[
    df["label"].isin([0, 1])
].copy()

df = df.drop_duplicates(
    subset=["url"]
).reset_index(drop=True)


print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)


X = df["url"]
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
    len(X_train),
)

print(
    "Validation records:",
    len(X_validation),
)

print(
    "Testing records   :",
    len(X_test),
)


print(
    "\nCreating character-level TF-IDF features..."
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    max_features=50000,
    min_df=2,
    sublinear_tf=True,
)

X_train_text = vectorizer.fit_transform(
    X_train
)

X_validation_text = vectorizer.transform(
    X_validation
)

X_test_text = vectorizer.transform(
    X_test
)


print(
    "TF-IDF features:",
    X_train_text.shape[1]
)


print(
    "\nCreating lexical URL features..."
)

X_train_lexical = extract_feature_matrix(
    X_train
)

X_validation_lexical = extract_feature_matrix(
    X_validation
)

X_test_lexical = extract_feature_matrix(
    X_test
)


print(
    "Lexical features:",
    X_train_lexical.shape[1]
)


scaler = StandardScaler()

X_train_lexical_scaled = scaler.fit_transform(
    X_train_lexical
)

X_validation_lexical_scaled = scaler.transform(
    X_validation_lexical
)

X_test_lexical_scaled = scaler.transform(
    X_test_lexical
)


X_train_combined = hstack(
    [
        X_train_text,
        csr_matrix(
            X_train_lexical_scaled
        ),
    ],
    format="csr",
)

X_validation_combined = hstack(
    [
        X_validation_text,
        csr_matrix(
            X_validation_lexical_scaled
        ),
    ],
    format="csr",
)

X_test_combined = hstack(
    [
        X_test_text,
        csr_matrix(
            X_test_lexical_scaled
        ),
    ],
    format="csr",
)


print(
    "\nTotal combined features:",
    X_train_combined.shape[1]
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        random_state=42,
        solver="liblinear",
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=50,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
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
    print(
        f"Training {model_name}"
    )
    print("=" * 70)

    start_time = time.time()

    current_model.fit(
        X_train_combined,
        y_train,
    )

    training_time = (
        time.time() - start_time
    )

    validation_predictions = (
        current_model.predict(
            X_validation_combined
        )
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions,
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

        best_validation_f1 = (
            validation_f1
        )

        best_model = (
            current_model
        )

        best_model_name = (
            model_name
        )


results_dataframe = (
    pd.DataFrame(
        results
    )
)

results_dataframe = (
    results_dataframe
    .sort_values(
        by="Validation F1",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 70)
print(
    "URL MODEL VERSION 3 VALIDATION COMPARISON"
)
print("=" * 70)

print(
    results_dataframe.to_string(
        index=False
    )
)


print("\n" + "=" * 70)

print(
    f"Best Model: "
    f"{best_model_name}"
)

print(
    f"Best Validation F1-score: "
    f"{best_validation_f1:.4f}"
)

print("=" * 70)


print(
    "\nEvaluating best model "
    "on final test set..."
)

test_predictions = (
    best_model.predict(
        X_test_combined
    )
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
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
    f"\nAccuracy  : "
    f"{test_accuracy:.4f}"
)

print(
    f"Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"F1-score  : "
    f"{test_f1:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Legitimate",
            "Malicious",
        ],
        zero_division=0,
    )
)


print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        test_predictions,
    )
)


joblib.dump(
    best_model,
    MODEL_PATH,
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH,
)

joblib.dump(
    scaler,
    SCALER_PATH,
)

results_dataframe.to_csv(
    RESULTS_PATH,
    index=False,
)


print(
    "\nSaved successfully:"
)

print(
    f"- {MODEL_PATH}"
)

print(
    f"- {VECTORIZER_PATH}"
)

print(
    f"- {SCALER_PATH}"
)

print(
    f"- {RESULTS_PATH}"
)
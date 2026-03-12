"""Train and save the transaction classification model."""

import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Path to save model artifacts
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")


def prepare_text(details: str) -> str:
    """Prepare details text as feature for classification."""
    details = str(details) if pd.notna(details) else ""
    return details.strip()


def train_model(data_path: str = None) -> Tuple[float, float]:
    """
    Train the classification model on labeled data.

    Args:
        data_path: Path to labeled CSV. Defaults to data/bank_statements_labeled.csv

    Returns:
        Tuple of (accuracy, f1_macro)
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "bank_statements_labeled.csv"
        )

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Prepare features
    df["text"] = df.apply(
        lambda row: prepare_text(row.get("details")),
        axis=1
    )

    # Filter out rows with empty text or missing category
    df = df[df["text"].str.strip() != ""]
    df = df[df["category"].notna()]

    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")

    X = df["text"].values
    y = df["category"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Vectorize
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        lowercase=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Features: {X_train_vec.shape[1]}")

    # Train model
    print("Training LinearSVC...")
    model = LinearSVC(random_state=42, max_iter=2000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save model and vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    return accuracy, f1


if __name__ == "__main__":
    train_model()

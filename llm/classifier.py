"""Transaction classifier using trained LinearSVC model."""

import os
import pickle
from typing import Optional, Dict, List

# Paths to model artifacts
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# Global model cache
_model = None
_vectorizer = None


def _load_model():
    """Load the trained model and vectorizer."""
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return True

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(f"Model not found. Run: python -m llm.model_trainer")
        return False

    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def classify_transaction(details: Optional[str]) -> str:
    """
    Classify a transaction into a category using the trained model.

    Args:
        details: Transaction details/description

    Returns:
        Category string (e.g., "переводы", "покупки", "транспорт", etc.)
    """
    # Prepare text
    text = str(details).strip() if details else ""

    if not text:
        return "другое"

    # Try to use trained model
    if _load_model():
        try:
            text_vec = _vectorizer.transform([text])
            prediction = _model.predict(text_vec)[0]
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")

    # Fallback to keyword-based classification
    return _keyword_classify(text)


def _keyword_classify(text: str) -> str:
    """Fallback keyword-based classification."""
    text_lower = text.lower()

    keywords = {
        "переводы": ["перевод", "transfer", "p2p", "на карту", "депозит", "кредит"],
        "транспорт": ["такси", "taxi", "yandex.go", "bolt", "uber", "avtobys", "азс", "бензин", "parking"],
        "рестораны и кафе": ["cafe", "кафе", "restaurant", "ресторан", "coffee", "еда", "food", "burger"],
        "супермаркеты": ["magnum", "small", "galmart", "supermarket", "супермаркет", "market"],
        "подписки": ["spotify", "netflix", "apple", "google", "subscription", "подписка"],
        "маркетплейсы": ["kaspi магазин", "wildberries", "ozon", "aliexpress"],
    }

    for category, kws in keywords.items():
        if any(kw in text_lower for kw in kws):
            return category

    return "покупки"


def classify_transactions(transactions: List[Dict]) -> List[Dict]:
    """
    Classify a list of transactions.

    Args:
        transactions: List of transaction dicts with 'details' field

    Returns:
        Same list with 'category' field added to each transaction
    """
    # Pre-load model once
    _load_model()

    for tx in transactions:
        tx["category"] = classify_transaction(
            tx.get("details")
        )
    return transactions

"""Transaction parsing and classification routes."""

import os
import tempfile

from flask import Blueprint, request, jsonify

from llm import parse_transactions, classify_transactions

transactions_bp = Blueprint("transactions", __name__)


@transactions_bp.route("/classify", methods=["POST"])
def classify():
    """
    Receive a PDF file, parse transactions, and return them categorized.

    Usage:
        curl -X POST -F "file=@statement.pdf" http://localhost:5001/classify

    Returns:
        {
            "bank": "Kaspi Bank",
            "transactions": [
                {
                    "date": "2024-11-13",
                    "amount": -1500.0,
                    "currency": "KZT",
                    "operation": "Покупка",
                    "details": "YANDEX.GO ALMATY KZ",
                    "category": "Transport"
                },
                ...
            ],
            "summary": {
                "total_transactions": 50,
                "by_category": {
                    "Transport": 10,
                    "Eating out": 15,
                    ...
                }
            }
        }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    # Save to temp file and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Parse transactions from PDF
        result = parse_transactions(tmp_path)

        if result.get("error"):
            return jsonify({
                "bank": result.get("bank"),
                "error": result["error"],
                "transactions": []
            }), 400 if not result.get("bank") else 200

        # Classify transactions
        transactions = classify_transactions(result["transactions"])

        # Build summary
        category_counts = {}
        for tx in transactions:
            cat = tx.get("category", "Other")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return jsonify({
            "bank": result["bank"],
            "transactions": transactions,
            "summary": {
                "total_transactions": len(transactions),
                "by_category": category_counts
            }
        })

    finally:
        os.unlink(tmp_path)

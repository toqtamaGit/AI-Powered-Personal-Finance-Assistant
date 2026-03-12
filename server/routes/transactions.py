"""Transaction parsing and classification routes."""

import os
import tempfile

from flask import Blueprint, request, jsonify

from categorize import process_pdf

transactions_bp = Blueprint("transactions", __name__)


@transactions_bp.route("/classify", methods=["POST"])
def classify():
    """
    Receive a PDF file, run the full categorization pipeline, and return
    enriched transactions as JSON.

    Usage:
        curl -X POST -F "file=@statement.pdf" http://localhost:5001/classify
        curl -X POST -F "file=@statement.pdf" "http://localhost:5001/classify?skip_oked=1"

    Query params (optional):
        skip_oked=1     - skip OKED code fetching (faster)
        skip_classify=1 - skip ML classification

    Returns:
        {
            "bank": "Kaspi Bank",
            "transactions": [
                {
                    "date": "2024-11-13",
                    "amount": -1500.0,
                    "currency": "KZT",
                    "operation": "Purchases",
                    "details": "YANDEX GO ALMATY",
                    "business_name": "YANDEX GO",
                    "city": "ALMATY",
                    "oked_description": "...",
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

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        skip_oked = request.args.get("skip_oked", "0") == "1"
        skip_classify = request.args.get("skip_classify", "0") == "1"

        df = process_pdf(
            tmp_path,
            workers=3,
            skip_oked=skip_oked,
            skip_classify=skip_classify,
        )

        # Bank name from the DataFrame
        bank_name = df["bank"].iloc[0] if "bank" in df.columns and len(df) else None

        # Replace NaN/NaT with None for clean JSON serialization
        df = df.where(df.notna(), None)

        # Convert to list of dicts
        transactions = df.to_dict(orient="records")

        # Build category summary
        category_counts = {}
        if "category" in df.columns:
            for cat in df["category"].dropna():
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return jsonify({
            "bank": bank_name,
            "transactions": transactions,
            "summary": {
                "total_transactions": len(transactions),
                "by_category": category_counts,
            },
        })

    except ValueError as e:
        return jsonify({"error": str(e), "transactions": []}), 400
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}", "transactions": []}), 500
    finally:
        os.unlink(tmp_path)

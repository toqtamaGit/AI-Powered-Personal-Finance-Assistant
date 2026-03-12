"""Bank detection routes."""

import os
import tempfile

from flask import Blueprint, request, jsonify

from llm import detect_bank_from_text, extract_text_from_pdf

bank_bp = Blueprint("bank", __name__)


@bank_bp.route("/detect-bank", methods=["POST"])
def detect_bank():
    """
    Receive a PDF file and return the detected bank name.

    Usage:
        curl -X POST -F "file=@statement.pdf" http://localhost:5000/detect-bank
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
        text = extract_text_from_pdf(tmp_path)
        bank_name = detect_bank_from_text(text)

        return jsonify({
            "bank": bank_name,
            "detected": bank_name is not None
        })
    finally:
        os.unlink(tmp_path)

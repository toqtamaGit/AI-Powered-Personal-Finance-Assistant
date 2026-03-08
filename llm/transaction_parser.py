"""Transaction parser — dispatches to the correct bank parser via the registry.

Kept for backward compatibility with server code.  New code should use
``categorize.py`` or the ``parsers.REGISTRY`` directly.
"""

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.bank_detector import detect_bank_from_text
from llm.pdf_extractor import extract_text_from_pdf
from parsers import REGISTRY


def parse_transactions(pdf_path: str) -> Dict:
    """
    Parse transactions from a bank statement PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dict with:
            - bank: Detected bank name
            - transactions: List of transaction dicts
            - error: Error message if parsing failed
    """
    text = extract_text_from_pdf(pdf_path)
    bank_name = detect_bank_from_text(text)

    if not bank_name:
        return {
            "bank": None,
            "transactions": [],
            "error": "Could not detect bank from PDF",
        }

    if bank_name not in REGISTRY:
        return {
            "bank": bank_name,
            "transactions": [],
            "error": f"No parser available for {bank_name}",
        }

    try:
        parser_cls, _ = REGISTRY[bank_name]
        parser = parser_cls()
        transactions, rejects = parser.parse(pdf_path)

        result: Dict = {
            "bank": bank_name,
            "transactions": transactions,
            "error": None,
        }
        if rejects:
            result["rejected_lines"] = len(rejects)
        return result

    except Exception as e:
        return {
            "bank": bank_name,
            "transactions": [],
            "error": str(e),
        }

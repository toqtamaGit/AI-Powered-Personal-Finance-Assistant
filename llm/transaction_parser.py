"""Transaction parser - extracts transactions from bank statement PDFs."""

import os
import sys
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.bank_detector import detect_bank_from_text
from llm.pdf_extractor import extract_text_from_pdf


def parse_kaspi_transactions(pdf_path: str) -> List[Dict]:
    """Parse transactions from Kaspi Bank PDF statement."""
    from parsers.kaspi import parse_kaspi_pdf
    return parse_kaspi_pdf(pdf_path)


def parse_freedom_transactions(pdf_path: str) -> Tuple[List[Dict], List[str]]:
    """Parse transactions from Freedom Bank PDF statement."""
    from parsers.freedom import parse_pdf
    return parse_pdf(pdf_path)


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
    # Extract text and detect bank
    text = extract_text_from_pdf(pdf_path)
    bank_name = detect_bank_from_text(text)

    if not bank_name:
        return {
            "bank": None,
            "transactions": [],
            "error": "Could not detect bank from PDF"
        }

    try:
        if bank_name == "Kaspi Bank":
            transactions = parse_kaspi_transactions(pdf_path)
            return {
                "bank": bank_name,
                "transactions": transactions,
                "error": None
            }

        elif bank_name == "Freedom Bank Kazakhstan":
            transactions, rejects = parse_freedom_transactions(pdf_path)
            return {
                "bank": bank_name,
                "transactions": transactions,
                "rejected_lines": len(rejects),
                "error": None
            }

        elif bank_name in ["Halyk Bank", "Jusan Bank"]:
            # For now, try Freedom parser as fallback (similar format)
            try:
                transactions, rejects = parse_freedom_transactions(pdf_path)
                return {
                    "bank": bank_name,
                    "transactions": transactions,
                    "rejected_lines": len(rejects),
                    "error": None
                }
            except Exception:
                return {
                    "bank": bank_name,
                    "transactions": [],
                    "error": f"Parser for {bank_name} not fully implemented yet"
                }

        else:
            return {
                "bank": bank_name,
                "transactions": [],
                "error": f"No parser available for {bank_name}"
            }

    except Exception as e:
        return {
            "bank": bank_name,
            "transactions": [],
            "error": str(e)
        }

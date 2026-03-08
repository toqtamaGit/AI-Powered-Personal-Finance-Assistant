"""
E2E tests — full pipeline including OKED fetching (requires network).

Run:  python -m pytest tests/e2e/ -v
Skip: python -m pytest tests/ -m "not e2e"
"""
import os

import pandas as pd
import pytest

from categorize import process_pdf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pdfs")
FREEDOM_PDF = os.path.join(DATA_DIR, "freedom", "freedom_ru.pdf")

SKIP_NO_PDF = not os.path.isfile(FREEDOM_PDF)

ALL_EXPECTED_COLS = [
    "date", "amount_raw", "amount", "currency", "operation", "details",
    "bank", "source_file",
    "business_type", "business_name", "address", "city",
    "oked_code", "oked_description",
    "category",
]


@pytest.mark.e2e
@pytest.mark.skipif(SKIP_NO_PDF, reason="PDF test files not available")
class TestFullPipeline:
    """Full pipeline: PDF → parse → normalize → OKED → classify."""

    def test_all_columns_present(self):
        df = process_pdf(FREEDOM_PDF, workers=3, skip_oked=False, skip_classify=False)
        for col in ALL_EXPECTED_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_oked_coverage(self):
        df = process_pdf(FREEDOM_PDF, workers=3, skip_oked=False, skip_classify=True)
        biz = df[df["business_name"].notna()]
        oked_filled = biz["oked_description"].notna().sum()
        coverage = oked_filled / len(biz) if len(biz) else 0
        assert coverage > 0.5, f"OKED coverage too low: {coverage:.0%}"

    def test_all_rows_classified(self):
        df = process_pdf(FREEDOM_PDF, workers=3, skip_oked=True, skip_classify=False)
        assert df["category"].notna().all()

    def test_no_standalone_digits(self):
        df = process_pdf(FREEDOM_PDF, workers=3, skip_oked=True, skip_classify=True)
        for name in df["business_name"].dropna():
            for token in name.split():
                assert not token.isdigit(), f"Standalone digit in: '{name}'"

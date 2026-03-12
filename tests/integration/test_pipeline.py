"""
Integration tests for the categorization pipeline — reads real PDFs, no network.

Run:  python -m pytest tests/integration/test_pipeline.py -v
"""

import os

import pandas as pd
import pytest

from categorize import process_pdf, collect_pdfs, detect_bank

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pdfs")
KASPI_PDF = os.path.join(DATA_DIR, "kaspi", "kaspi_en.pdf")
FREEDOM_PDF = os.path.join(DATA_DIR, "freedom", "freedom_en.pdf")

SKIP_REASON = "PDF test files not available"

# Expected columns in the final output
BASE_COLS = ["date", "amount_raw", "amount", "currency", "operation", "details", "bank", "source_file"]
ENRICHED_COLS = ["business_type", "business_name", "address", "city"]
OKED_COLS = ["oked_code", "oked_description"]
CLASSIFY_COLS = ["category"]
ALL_COLS = BASE_COLS + ENRICHED_COLS


# ---------------------------------------------------------------------------
# collect_pdfs
# ---------------------------------------------------------------------------

class TestCollectPdfs:
    @pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason=SKIP_REASON)
    def test_collect_from_directory(self):
        pdfs = collect_pdfs(DATA_DIR)
        assert len(pdfs) >= 1
        assert all(p.endswith(".pdf") for p in pdfs)

    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_collect_single_file(self):
        pdfs = collect_pdfs(KASPI_PDF)
        assert len(pdfs) == 1

    def test_invalid_path(self):
        with pytest.raises(SystemExit):
            collect_pdfs("/nonexistent/path.pdf")


# ---------------------------------------------------------------------------
# detect_bank
# ---------------------------------------------------------------------------

class TestDetectBank:
    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_detect_kaspi(self):
        bank = detect_bank(KASPI_PDF)
        assert bank == "Kaspi Bank"

    @pytest.mark.skipif(not os.path.isfile(FREEDOM_PDF), reason=SKIP_REASON)
    def test_detect_freedom(self):
        bank = detect_bank(FREEDOM_PDF)
        assert bank == "Freedom Bank Kazakhstan"


# ---------------------------------------------------------------------------
# process_pdf — Kaspi
# ---------------------------------------------------------------------------

class TestProcessPdfKaspi:
    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_returns_dataframe(self):
        df = process_pdf(KASPI_PDF, skip_oked=True, skip_classify=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_has_base_columns(self):
        df = process_pdf(KASPI_PDF, skip_oked=True, skip_classify=True)
        for col in ALL_COLS:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_bank_field(self):
        df = process_pdf(KASPI_PDF, skip_oked=True, skip_classify=True)
        assert (df["bank"] == "Kaspi Bank").all()

    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_amounts_are_numeric(self):
        df = process_pdf(KASPI_PDF, skip_oked=True, skip_classify=True)
        assert df["amount"].dtype in ("float64", "int64")

    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_dates_are_valid(self):
        df = process_pdf(KASPI_PDF, skip_oked=True, skip_classify=True)
        dates = pd.to_datetime(df["date"], errors="coerce")
        assert dates.notna().all(), "Some dates could not be parsed"

    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_with_classify(self):
        df = process_pdf(KASPI_PDF, skip_oked=True, skip_classify=False)
        assert "category" in df.columns
        assert df["category"].notna().all()

    @pytest.mark.skipif(not os.path.isfile(KASPI_PDF), reason=SKIP_REASON)
    def test_no_standalone_digits_in_business_name(self):
        df = process_pdf(KASPI_PDF, skip_oked=True, skip_classify=True)
        bnames = df["business_name"].dropna()
        for name in bnames:
            for token in name.split():
                assert not token.isdigit(), f"Standalone digit in business_name: '{name}'"


# ---------------------------------------------------------------------------
# process_pdf — Freedom
# ---------------------------------------------------------------------------

class TestProcessPdfFreedom:
    @pytest.mark.skipif(not os.path.isfile(FREEDOM_PDF), reason=SKIP_REASON)
    def test_returns_dataframe(self):
        df = process_pdf(FREEDOM_PDF, skip_oked=True, skip_classify=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.skipif(not os.path.isfile(FREEDOM_PDF), reason=SKIP_REASON)
    def test_has_base_columns(self):
        df = process_pdf(FREEDOM_PDF, skip_oked=True, skip_classify=True)
        for col in ALL_COLS:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.skipif(not os.path.isfile(FREEDOM_PDF), reason=SKIP_REASON)
    def test_bank_field(self):
        df = process_pdf(FREEDOM_PDF, skip_oked=True, skip_classify=True)
        assert (df["bank"] == "Freedom Bank Kazakhstan").all()

    @pytest.mark.skipif(not os.path.isfile(FREEDOM_PDF), reason=SKIP_REASON)
    def test_has_country_column(self):
        """Freedom normalizer adds a 'country' field."""
        df = process_pdf(FREEDOM_PDF, skip_oked=True, skip_classify=True)
        assert "country" in df.columns

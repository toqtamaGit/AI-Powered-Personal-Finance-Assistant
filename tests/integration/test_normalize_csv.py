"""
Integration tests for normalize_csv — reads/writes temp CSV files.

Run:  python -m pytest tests/integration/test_normalize_csv.py -v
"""
import os
import tempfile

import pandas as pd
import pytest

from parsers.normalize_kaspi import normalize_csv as kaspi_normalize_csv
from parsers.normalize_freedom import normalize_csv as freedom_normalize_csv


# ---------------------------------------------------------------------------
# Kaspi: deduplication via normalize_csv
# ---------------------------------------------------------------------------

class TestKaspiDeduplication:
    def _run(self, rows):
        df_in = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.csv")
            out = os.path.join(td, "out.csv")
            df_in.to_csv(inp, index=False)
            return kaspi_normalize_csv(inp, out)

    def test_duplicates_removed(self):
        rows = [
            {"date": "2024-11-13", "amount": -100, "operation": "Purchases",
             "details": "Magnum Cash&Carry"},
            {"date": "2024-11-14", "amount": -200, "operation": "Purchases",
             "details": "Magnum Cash&Carry"},
            {"date": "2024-11-15", "amount": -300, "operation": "Purchases",
             "details": "Green Food"},
        ]
        df = self._run(rows)
        assert len(df) == 2
        assert df.iloc[0]["date"] == "2024-11-13"

    def test_different_details_kept(self):
        rows = [
            {"date": "2024-11-13", "amount": -100, "operation": "Purchases",
             "details": "Magnum Cash&Carry"},
            {"date": "2024-11-14", "amount": -200, "operation": "Purchases",
             "details": "Green Food"},
            {"date": "2024-11-15", "amount": -300, "operation": "Transfers",
             "details": "Саида О."},
        ]
        df = self._run(rows)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Kaspi: normalize_csv output columns
# ---------------------------------------------------------------------------

class TestKaspiNormalizeCsvOutput:
    def test_output_has_all_columns(self):
        rows = [
            {"date": "2024-11-13", "amount": -100, "operation": "Purchases",
             "details": "ИП Берсиновв"},
            {"date": "2024-11-14", "amount": -200, "operation": "Transfers",
             "details": "Саида О."},
        ]
        df_in = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.csv")
            out = os.path.join(td, "out.csv")
            df_in.to_csv(inp, index=False)
            df_out = kaspi_normalize_csv(inp, out)

        assert "business_type" in df_out.columns
        assert "business_name" in df_out.columns
        assert "address" in df_out.columns
        assert "city" in df_out.columns
        assert df_out.iloc[0]["business_type"] == "IP"
        assert df_out.iloc[0]["business_name"] == "Берсиновв"
        assert pd.isna(df_out.iloc[1]["business_type"])
        assert pd.isna(df_out.iloc[1]["business_name"])

    def test_city_column_populated(self):
        rows = [
            {"date": "2024-11-13", "amount": -100, "operation": "Purchases",
             "details": "Рестопарк AVATARIYA Тараз"},
        ]
        df_in = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.csv")
            out = os.path.join(td, "out.csv")
            df_in.to_csv(inp, index=False)
            df_out = kaspi_normalize_csv(inp, out)

        assert df_out.iloc[0]["city"] == "TARAZ"
        assert df_out.iloc[0]["business_name"] == "Рестопарк AVATARIYA"


# ---------------------------------------------------------------------------
# Freedom: deduplication via normalize_csv
# ---------------------------------------------------------------------------

class TestFreedomDeduplication:
    def _run(self, rows):
        df_in = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.csv")
            out = os.path.join(td, "out.csv")
            df_in.to_csv(inp, index=False)
            return freedom_normalize_csv(inp, out)

    def test_duplicates_removed(self):
        rows = [
            {"date": "2025-01-01", "amount": -100, "operation": "Acquiring",
             "details": "IP GLOBAL MANAGEMENT ASTANA KZ"},
            {"date": "2025-01-02", "amount": -200, "operation": "Acquiring",
             "details": "IP GLOBAL MANAGEMENT ASTANA KZ"},
            {"date": "2025-01-03", "amount": -300, "operation": "Acquiring",
             "details": "YANDEX.EDA ALMATY KZ"},
        ]
        df = self._run(rows)
        assert len(df) == 2
        assert df.iloc[0]["date"] == "2025-01-01"

    def test_different_details_kept(self):
        rows = [
            {"date": "2025-01-01", "amount": -100, "operation": "Acquiring",
             "details": "IP GLOBAL MANAGEMENT ASTANA KZ"},
            {"date": "2025-01-02", "amount": -200, "operation": "Acquiring",
             "details": "YANDEX.EDA ALMATY KZ"},
            {"date": "2025-01-03", "amount": -300, "operation": "Transfer",
             "details": "Перевод с карты на карту"},
        ]
        df = self._run(rows)
        assert len(df) == 3

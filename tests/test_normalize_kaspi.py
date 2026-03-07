"""
Tests for llm/normalize_kaspi.py — Kaspi CSV normalization.

Run:  python -m pytest tests/test_normalize_kaspi.py -v
"""
import os
import tempfile

import pandas as pd
import pytest

from llm.normalize_kaspi import parse_details, normalize_csv


# ---------------------------------------------------------------------------
#  1. parse_details — business type extraction
# ---------------------------------------------------------------------------

class TestParseDetails:
    def test_ip_cyrillic(self):
        btype, bname = parse_details("ИП Берсиновв", "Purchases")
        assert btype == "IP"
        assert bname == "Берсиновв"

    def test_too_cyrillic(self):
        btype, bname = parse_details("ТОО Kaspi Travel", "Purchases")
        assert btype == "TOO"
        assert bname == "Kaspi Travel"

    def test_too_latin(self):
        btype, bname = parse_details('TOO "Kaspi Magazin"', "Purchases")
        assert btype == "TOO"
        assert bname == "Kaspi Magazin"

    def test_ip_latin(self):
        btype, bname = parse_details("IP R TRADE", "Purchases")
        assert btype == "IP"
        assert bname == "R TRADE"

    def test_ip_lowercase(self):
        btype, bname = parse_details("ип маденова", "Purchases")
        assert btype == "IP"
        assert bname == "маденова"

    def test_too_suffix(self):
        btype, bname = parse_details("Малина ТОО", "Purchases")
        assert btype == "TOO"
        assert bname == "Малина"

    def test_too_middle(self):
        btype, bname = parse_details('Филиал ТОО "Гелиос" в городе Астана АЗС 14', "Purchases")
        assert btype == "TOO"
        assert "Гелиос" in bname

    def test_llc_middle(self):
        btype, bname = parse_details("ADVENTURE HQ LLC (- 120,00 AED)", "Purchases")
        assert btype == "LLC"
        assert "ADVENTURE HQ" in bname

    def test_no_type(self):
        btype, bname = parse_details("Magnum Cash&Carry", "Purchases")
        assert btype is None
        assert bname == "Magnum Cash&Carry"

    def test_no_type_simple(self):
        btype, bname = parse_details("Green Food", "Purchases")
        assert btype is None
        assert bname == "Green Food"

    def test_quoted_name(self):
        btype, bname = parse_details('ТОО "Книга-НВ"', "Purchases")
        assert btype == "TOO"
        assert bname == "Книга-НВ"

    def test_withdrawal(self):
        btype, bname = parse_details(
            "Банкомат Iliyasa Zhansugurova 8k1", "Withdrawal"
        )
        assert btype is None
        assert bname == "Банкомат Iliyasa Zhansugurova 8k1"

    def test_blocked_suffix_stripped(self):
        btype, bname = parse_details(
            "Magnum Cash&Carry - The amount is blocked. The bank expects the confirmation of the payment system.",
            "Purchases",
        )
        assert btype is None
        assert bname == "Magnum Cash&Carry"

    def test_blocked_suffix_russian(self):
        btype, bname = parse_details(
            "Ресторан Trattoria Highvill - Сумма заблокирована. Банк ожидает подтверждения.",
            "Purchases",
        )
        assert btype is None
        assert bname == "Ресторан Trattoria Highvill"


# ---------------------------------------------------------------------------
#  2. Non-purchase operations → all None
# ---------------------------------------------------------------------------

class TestNonPurchase:
    @pytest.mark.parametrize("op", [
        "Transfers", "Replenishment", "Others",
    ])
    def test_returns_none(self, op):
        btype, bname = parse_details("ТОО Kaspi Travel", op)
        assert btype is None
        assert bname is None


# ---------------------------------------------------------------------------
#  3. Deduplication by details column
# ---------------------------------------------------------------------------

class TestDeduplication:
    def _run(self, rows):
        df_in = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.csv")
            out = os.path.join(td, "out.csv")
            df_in.to_csv(inp, index=False)
            df_out = normalize_csv(inp, out)
        return df_out

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
#  4. normalize_csv produces correct columns
# ---------------------------------------------------------------------------

class TestNormalizeCsvOutput:
    def test_output_has_business_columns(self):
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
            df_out = normalize_csv(inp, out)

        assert "business_type" in df_out.columns
        assert "business_name" in df_out.columns
        assert df_out.iloc[0]["business_type"] == "IP"
        assert df_out.iloc[0]["business_name"] == "Берсиновв"
        assert pd.isna(df_out.iloc[1]["business_type"])
        assert pd.isna(df_out.iloc[1]["business_name"])

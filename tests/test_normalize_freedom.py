"""
Tests for llm/normalize_freedom.py — business_type, business_name, address, city, country extraction.

Run:  python -m pytest tests/test_normalize_freedom.py -v
"""
import os
import tempfile

import pandas as pd
import pytest

from llm.normalize_freedom import (
    parse_freedom_details,
    normalize_hyphens,
    normalize_csv,
    extract_city_from_text,
)


# ---------------------------------------------------------------------------
#  1. Standard pattern: TYPE NAME CITY COUNTRY
# ---------------------------------------------------------------------------

class TestStandardPattern:
    def test_ip_with_city_country(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "IP GLOBAL MANAGEMENT ASTANA KZ", "Acquiring"
        )
        assert btype == "IP"
        assert bname == "GLOBAL MANAGEMENT"
        assert city == "ASTANA"
        assert country == "KZ"

    def test_too_with_city_country(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "TOO UNIQLOAST ASTANA KZ", "Acquiring"
        )
        assert btype == "TOO"
        assert bname == "UNIQLOAST"
        assert city == "ASTANA"
        assert country == "KZ"

    def test_no_type_with_city_country(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "MOLA COFFEE HOUSE ASTANA KZ", "Acquiring"
        )
        assert btype is None
        assert bname == "MOLA COFFEE HOUSE"
        assert city == "ASTANA"
        assert country == "KZ"

    def test_almaty(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "YANDEX.EDA ALMATY KZ", "Acquiring"
        )
        assert btype is None
        assert bname == "YANDEX.EDA"
        assert city == "ALMATY"
        assert country == "KZ"

    def test_taraz(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "IP BUTYRSKAYA G.A. TARAZ KZ", "Acquiring"
        )
        assert btype == "IP"
        assert bname == "BUTYRSKAYA G.A."
        assert city == "TARAZ"
        assert country == "KZ"

    def test_pavlodar(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "IP DINARA PAVLODAR KZ", "Acquiring"
        )
        assert btype == "IP"
        assert bname == "DINARA"
        assert city == "PAVLODAR"
        assert country == "KZ"

    def test_foreign_country(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "THE BRITISH COUNCIL 441619577755 GB", "Acquiring"
        )
        assert btype is None
        assert bname == "THE BRITISH COUNCIL 441619577755"
        assert city is None
        assert country == "GB"

    def test_us_country(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "CLAUDE.AI SUBSCRIPTION 14152360599 US", "Acquiring"
        )
        assert btype is None
        assert bname == "CLAUDE.AI SUBSCRIPTION 14152360599"
        assert city is None
        assert country == "US"


# ---------------------------------------------------------------------------
#  2. City without country code
# ---------------------------------------------------------------------------

class TestCityNoCountry:
    def test_too_city_only(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "TOO GLASMAN TRADE 3 ASTANA", "Acquiring"
        )
        assert btype == "TOO"
        assert bname == "GLASMAN TRADE 3"
        assert city == "ASTANA"
        assert country is None

    def test_magnum_nur_sultan(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "MAGNUM CASH CARRY NF46 NUR-SULTAN", "Acquiring"
        )
        assert btype is None
        assert bname == "MAGNUM CASH CARRY NF46"
        assert city == "ASTANA"
        assert country is None


# ---------------------------------------------------------------------------
#  3. Q. suffix on city
# ---------------------------------------------------------------------------

class TestQSuffix:
    def test_astana_q_dot(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "AITMEN BARBERSHOP ASTANA Q. KZ", "Acquiring"
        )
        assert btype is None
        assert bname == "AITMEN BARBERSHOP"
        assert city == "ASTANA"
        assert country == "KZ"
        assert "Q." not in (bname or "")

    def test_ip_smagulova_q_dot(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "IP SMAGULOVA M ASTANA Q.", "Acquiring"
        )
        assert btype == "IP"
        assert bname == "SMAGULOVA M"
        assert city == "ASTANA"
        assert country is None
        assert "Q" not in (bname or "")

    def test_magazin_q_kz(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "MAGAZIN PRODUKTY ASTANA Q. KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "MAGAZIN PRODUKTY"

    def test_book_cafe_q_kz(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "BOOK CAFE ASTANA Q. KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "BOOK CAFE"


# ---------------------------------------------------------------------------
#  4. NUR-SULTAN → ASTANA normalization
# ---------------------------------------------------------------------------

class TestNurSultan:
    def test_nur_sultan_kz(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "OUTFIT NUR-SULTAN KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "OUTFIT"
        assert "NUR-SULTAN" not in (bname or "").upper()

    def test_hp_canteen_nur_sultan(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "HP CANTEEN NUR-SULTAN KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "HP CANTEEN"


# ---------------------------------------------------------------------------
#  5. Dual city — ASTANA Nur-Sultan
# ---------------------------------------------------------------------------

class TestDualCity:
    def test_tangiers_astana_nur_sultan(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "TANGIERS LOUNGE EC ASTANA Nur-Sultan", "Acquiring"
        )
        assert city == "ASTANA"
        assert "Nur-Sultan" not in (bname or "")
        assert "NUR-SULTAN" not in (bname or "").upper()

    def test_tlp_astana_nur_sultan_kz(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "TLP ASTANA Nur-Sultan KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert "NUR-SULTAN" not in (bname or "").upper()


# ---------------------------------------------------------------------------
#  6. No city/country (Payment with Russian text)
# ---------------------------------------------------------------------------

class TestNoCityCountry:
    def test_payment_russian(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "ТОО Aviata За оплату билета заказ 192433397", "Payment"
        )
        assert btype == "TOO"
        assert "Aviata" in bname
        assert city is None
        assert country is None

    def test_payment_arbuz(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "ТОО Arbuz Group Арбуз Груп За оплату билета заказ 3839164", "Payment"
        )
        assert btype == "TOO"
        assert "Arbuz" in bname
        assert city is None
        assert country is None

    def test_refund_no_location(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "Возврат. Отмена покупки", "Acquiring"
        )
        assert btype is None
        assert bname == "Возврат. Отмена покупки"
        assert city is None
        assert country is None


# ---------------------------------------------------------------------------
#  7. No city in text → city = None
# ---------------------------------------------------------------------------

class TestNoCity:
    def test_bersinova_momysh(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "IP BERSINOVA S.IM.B.MOMYSH KZ", "Acquiring"
        )
        assert btype == "IP"
        assert city is None
        assert country == "KZ"

    def test_kaan_no_city(self):
        btype, bname, addr, city, country = parse_freedom_details(
            'TOO K A A N', "Acquiring"
        )
        assert btype == "TOO"
        assert bname == "K A A N"
        assert addr is None
        assert city is None
        assert country is None

    def test_cursor_ai(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "CURSOR AI POWERED IDE 18314259504", "Acquiring"
        )
        assert city is None
        assert bname == "CURSOR AI POWERED IDE 18314259504"
        assert addr is None


# ---------------------------------------------------------------------------
#  8. Non-business operations return all None
# ---------------------------------------------------------------------------

class TestNonBusinessOps:
    @pytest.mark.parametrize("op", ["Replenishment", "Transfer", "Others", "Commission"])
    def test_skipped(self, op):
        btype, bname, addr, city, country = parse_freedom_details(
            "IP GLOBAL MANAGEMENT ASTANA KZ", op
        )
        assert btype is None
        assert bname is None
        assert addr is None
        assert city is None
        assert country is None

    def test_empty_details(self):
        btype, bname, addr, city, country = parse_freedom_details("", "Acquiring")
        assert btype is None
        assert bname is None
        assert city is None
        assert country is None

    def test_none_details(self):
        btype, bname, addr, city, country = parse_freedom_details(None, "Acquiring")
        assert btype is None
        assert bname is None
        assert city is None
        assert country is None


# ---------------------------------------------------------------------------
#  9. Known places move to address
# ---------------------------------------------------------------------------

class TestKnownPlaces:
    def test_mega_silk_way(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "ALYE PARUSA MEGA SILK WAY", "Acquiring"
        )
        assert bname == "ALYE PARUSA"
        assert "MEGA" in (addr or "")

    def test_burger_king_mega(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "BURGER KING MEGA SILK WAY", "Acquiring"
        )
        assert bname == "BURGER KING"
        assert "MEGA" in (addr or "")

    def test_dostyk_plaza(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "STARBUCKS DOSTYK PLAZA ALMATY KZ", "Acquiring"
        )
        assert bname == "STARBUCKS"
        assert "DOSTYK PLAZA" in (addr or "")
        assert city == "ALMATY"
        assert country == "KZ"


# ---------------------------------------------------------------------------
# 10. Pending amount operation
# ---------------------------------------------------------------------------

class TestPendingAmount:
    def test_pending_amount_parsed(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "TOO KIKIS MS NUR-SULTAN KZ", "Pending amount"
        )
        assert btype == "TOO"
        assert bname == "KIKIS MS"
        assert city == "ASTANA"
        assert country == "KZ"

    def test_pending_no_location(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "Перевод с карты на карту", "Pending amount"
        )
        assert btype is None
        assert bname == "Перевод с карты на карту"
        assert city is None
        assert country is None

    def test_pending_daniyarov(self):
        btype, bname, addr, city, country = parse_freedom_details(
            "IP DANIYAROV A ASTANA KZ", "Pending amount"
        )
        assert btype == "IP"
        assert bname == "DANIYAROV A"
        assert city == "ASTANA"


# ---------------------------------------------------------------------------
# 11. extract_city_from_text
# ---------------------------------------------------------------------------

class TestExtractCity:
    def test_astana(self):
        assert extract_city_from_text("ASTANA KZ") == "ASTANA"

    def test_nur_sultan_normalizes(self):
        assert extract_city_from_text("NUR-SULTAN KZ") == "ASTANA"

    def test_almaty(self):
        assert extract_city_from_text("ALMATY") == "ALMATY"

    def test_no_city(self):
        assert extract_city_from_text("S.IM.B.MOMYSH KZ") is None

    def test_empty(self):
        assert extract_city_from_text("") is None
        assert extract_city_from_text(None) is None


# ---------------------------------------------------------------------------
# 12. normalize_hyphens
# ---------------------------------------------------------------------------

class TestNormalizeHyphens:
    def test_space_after(self):
        assert normalize_hyphens("NUR- SULTAN") == "NUR-SULTAN"

    def test_space_before(self):
        assert normalize_hyphens("NUR -SULTAN") == "NUR-SULTAN"

    def test_en_dash(self):
        assert normalize_hyphens("NUR\u2013SULTAN") == "NUR-SULTAN"

    def test_em_dash(self):
        assert normalize_hyphens("NUR\u2014SULTAN") == "NUR-SULTAN"

    def test_none(self):
        assert normalize_hyphens(None) is None


# ---------------------------------------------------------------------------
# 13. Deduplication by details column
# ---------------------------------------------------------------------------

class TestDeduplication:
    """normalize_csv should drop rows with duplicate details, keeping first."""

    def _run(self, rows):
        """Helper: write rows to a temp CSV, run normalize_csv, return result df."""
        df_in = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.csv")
            out = os.path.join(td, "out.csv")
            df_in.to_csv(inp, index=False)
            df_out = normalize_csv(inp, out)
        return df_out

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

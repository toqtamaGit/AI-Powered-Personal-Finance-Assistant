"""
Tests for llm/normalize.py — business detail extraction from Freedom CSVs.

Run:  python -m pytest tests/test_normalize.py -v
"""
import pytest
from llm.normalize import (
    parse_details,
    normalize_hyphens,
    extract_city_from_text,
)


# ---------------------------------------------------------------------------
#  1. Standard patterns
# ---------------------------------------------------------------------------

class TestStandardPatterns:
    def test_too_with_city_kz(self):
        btype, bname, addr, city = parse_details(
            "TOO GLASMAN TRADE 3 ASTANA KZ", "Acquiring"
        )
        assert btype == "TOO"
        assert bname == "GLASMAN TRADE 3"
        assert city == "ASTANA"

    def test_ip_with_city_kz(self):
        btype, bname, addr, city = parse_details(
            "IP GLOBAL MANAGEMENT ASTANA KZ", "Acquiring"
        )
        assert btype == "IP"
        assert bname == "GLOBAL MANAGEMENT"
        assert city == "ASTANA"

    def test_no_type_with_city_kz(self):
        btype, bname, addr, city = parse_details(
            "MOLA COFFEE HOUSE ASTANA KZ", "Acquiring"
        )
        assert btype is None
        assert bname == "MOLA COFFEE HOUSE"
        assert city == "ASTANA"

    def test_city_without_kz(self):
        btype, bname, addr, city = parse_details(
            "TOO GLASMAN TRADE 3 ASTANA", "Acquiring"
        )
        assert btype == "TOO"
        assert city == "ASTANA"

    def test_almaty(self):
        btype, bname, addr, city = parse_details(
            "YANDEX.EDA ALMATY KZ", "Acquiring"
        )
        assert bname == "YANDEX.EDA"
        assert city == "ALMATY"

    def test_pavlodar(self):
        btype, bname, addr, city = parse_details(
            "IP DINARA PAVLODAR KZ", "Acquiring"
        )
        assert btype == "IP"
        assert bname == "DINARA"
        assert city == "PAVLODAR"


# ---------------------------------------------------------------------------
#  2. Q. district abbreviation — should be discarded
# ---------------------------------------------------------------------------

class TestQDiscard:
    def test_astana_q_kz(self):
        btype, bname, addr, city = parse_details(
            "AITMEN BARBERSHOP ASTANA Q. KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert "Q." not in (bname or "")
        assert bname == "AITMEN BARBERSHOP"

    def test_astana_q_no_kz(self):
        btype, bname, addr, city = parse_details(
            "IP SMAGULOVA M ASTANA Q.", "Acquiring"
        )
        assert btype == "IP"
        assert city == "ASTANA"
        assert "Q" not in (bname or "")

    def test_magazin_q_kz(self):
        btype, bname, addr, city = parse_details(
            "MAGAZIN PRODUKTY ASTANA Q. KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "MAGAZIN PRODUKTY"

    def test_book_cafe_q_kz(self):
        btype, bname, addr, city = parse_details(
            "BOOK CAFE ASTANA Q. KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "BOOK CAFE"


# ---------------------------------------------------------------------------
#  3. NUR-SULTAN → ASTANA
# ---------------------------------------------------------------------------

class TestNurSultan:
    def test_nur_sultan_kz(self):
        btype, bname, addr, city = parse_details(
            "OUTFIT NUR-SULTAN KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "OUTFIT"
        # NUR-SULTAN should not be in business name
        assert "NUR-SULTAN" not in (bname or "").upper()

    def test_hp_canteen_nur_sultan(self):
        btype, bname, addr, city = parse_details(
            "HP CANTEEN NUR-SULTAN KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert bname == "HP CANTEEN"


# ---------------------------------------------------------------------------
#  4. Dual city — ASTANA Nur-Sultan
# ---------------------------------------------------------------------------

class TestDualCity:
    def test_tangiers_astana_nur_sultan(self):
        btype, bname, addr, city = parse_details(
            "TANGIERS LOUNGE EC ASTANA Nur-Sultan", "Acquiring"
        )
        assert city == "ASTANA"
        # Nur-Sultan should NOT be in business name
        assert "Nur-Sultan" not in (bname or "")
        assert "NUR-SULTAN" not in (bname or "").upper()

    def test_tlp_astana_nur_sultan_kz(self):
        btype, bname, addr, city = parse_details(
            "TLP ASTANA Nur-Sultan KZ", "Acquiring"
        )
        assert city == "ASTANA"
        assert "NUR-SULTAN" not in (bname or "").upper()


# ---------------------------------------------------------------------------
#  5. No city in text → city = None
# ---------------------------------------------------------------------------

class TestNoCity:
    def test_bersinova_momysh(self):
        btype, bname, addr, city = parse_details(
            "IP BERSINOVA S.IM.B.MOMYSH KZ", "Acquiring"
        )
        assert btype == "IP"
        assert city is None

    def test_kaan_no_city(self):
        # Quotes already stripped by parser before reaching normalize
        btype, bname, addr, city = parse_details(
            'TOO K A A N', "Acquiring"
        )
        assert btype == "TOO"
        assert bname == "K A A N"
        assert addr is None
        assert city is None

    def test_cursor_ai(self):
        btype, bname, addr, city = parse_details(
            "CURSOR AI POWERED IDE 18314259504", "Acquiring"
        )
        assert city is None
        assert bname == "CURSOR AI POWERED IDE 18314259504"
        assert addr is None

    def test_claude_ai_us(self):
        btype, bname, addr, city = parse_details(
            "CLAUDE.AI SUBSCRIPTION 14152360599 US", "Acquiring"
        )
        assert city is None

    def test_british_council_gb(self):
        btype, bname, addr, city = parse_details(
            "THE BRITISH COUNCIL 441619577755 GB", "Acquiring"
        )
        assert city is None


# ---------------------------------------------------------------------------
#  6. Non-Acquiring operations → all None
# ---------------------------------------------------------------------------

class TestNonAcquiring:
    @pytest.mark.parametrize("op", [
        "Replenishment", "Transfer", "Others", "Payment", "Commission",
    ])
    def test_returns_none(self, op):
        btype, bname, addr, city = parse_details(
            "TOO GLASMAN TRADE 3 ASTANA KZ", op
        )
        assert btype is None
        assert bname is None
        assert addr is None
        assert city is None


# ---------------------------------------------------------------------------
#  7. Known places move to address
# ---------------------------------------------------------------------------

class TestKnownPlaces:
    def test_mega_silk_way(self):
        btype, bname, addr, city = parse_details(
            "ALYE PARUSA MEGA SILK WAY", "Acquiring"
        )
        assert bname == "ALYE PARUSA"
        assert "MEGA" in (addr or "")

    def test_burger_king_mega(self):
        btype, bname, addr, city = parse_details(
            "BURGER KING MEGA SILK WAY", "Acquiring"
        )
        assert bname == "BURGER KING"


# ---------------------------------------------------------------------------
#  8. Quotes in business names
# ---------------------------------------------------------------------------

class TestQuotes:
    """Quotes are now stripped by the parser. Normalizer receives clean text."""

    def test_quoted_name_already_stripped(self):
        # Parser strips quotes; normalizer gets clean input
        btype, bname, addr, city = parse_details(
            'TOO UNIQLOAST ASTANA KZ', "Acquiring"
        )
        assert btype == "TOO"
        assert bname == "UNIQLOAST"
        assert city == "ASTANA"

    def test_double_quoted_already_stripped(self):
        btype, bname, addr, city = parse_details(
            'TOO KIKIS MS ASTANA KZ', "Acquiring"
        )
        assert btype == "TOO"
        assert "KIKIS" in (bname or "")
        assert city == "ASTANA"


# ---------------------------------------------------------------------------
#  9. Pending amount works the same
# ---------------------------------------------------------------------------

class TestPendingAmount:
    def test_pending_amount(self):
        btype, bname, addr, city = parse_details(
            "IP DANIYAROV A ASTANA KZ", "Pending amount"
        )
        assert btype == "IP"
        assert bname == "DANIYAROV A"
        assert city == "ASTANA"


# ---------------------------------------------------------------------------
# 10. extract_city_from_text
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
# 11. normalize_hyphens
# ---------------------------------------------------------------------------

class TestNormalizeHyphens:
    def test_space_after(self):
        assert normalize_hyphens("NUR- SULTAN") == "NUR-SULTAN"

    def test_space_before(self):
        assert normalize_hyphens("NUR -SULTAN") == "NUR-SULTAN"

    def test_none(self):
        assert normalize_hyphens(None) is None

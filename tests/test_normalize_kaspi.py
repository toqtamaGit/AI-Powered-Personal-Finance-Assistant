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
        btype, bname, addr, city = parse_details("ИП Берсиновв", "Purchases")
        assert btype == "IP"
        assert bname == "Берсиновв"

    def test_too_cyrillic(self):
        btype, bname, addr, city = parse_details("ТОО Kaspi Travel", "Purchases")
        assert btype == "TOO"
        assert bname == "Kaspi Travel"

    def test_too_latin(self):
        btype, bname, addr, city = parse_details('TOO "Kaspi Magazin"', "Purchases")
        assert btype == "TOO"
        assert bname == "Kaspi Magazin"

    def test_ip_latin(self):
        btype, bname, addr, city = parse_details("IP R TRADE", "Purchases")
        assert btype == "IP"
        assert bname == "R TRADE"

    def test_ip_lowercase(self):
        btype, bname, addr, city = parse_details("ип маденова", "Purchases")
        assert btype == "IP"
        assert bname == "маденова"

    def test_too_suffix(self):
        btype, bname, addr, city = parse_details("Малина ТОО", "Purchases")
        assert btype == "TOO"
        assert bname == "Малина"

    def test_too_middle(self):
        btype, bname, addr, city = parse_details(
            'Филиал ТОО Гелиос в городе Астана АЗС 14', "Purchases"
        )
        assert btype == "TOO"
        assert "Гелиос" in bname

    def test_llc_middle(self):
        btype, bname, addr, city = parse_details(
            "ADVENTURE HQ LLC (- 120,00 AED)", "Purchases"
        )
        assert btype == "LLC"
        assert "ADVENTURE HQ" in bname

    def test_no_type(self):
        btype, bname, addr, city = parse_details("Magnum Cash&Carry", "Purchases")
        assert btype is None
        assert bname == "Magnum Cash&Carry"

    def test_no_type_simple(self):
        btype, bname, addr, city = parse_details("Green Food", "Purchases")
        assert btype is None
        assert bname == "Green Food"

    def test_quoted_name(self):
        btype, bname, addr, city = parse_details('ТОО "Книга-НВ"', "Purchases")
        assert btype == "TOO"
        assert bname == "Книга-НВ"

    def test_comma_stripped(self):
        btype, bname, addr, city = parse_details(
            "Bambino, Керуен", "Purchases"
        )
        assert "," not in (bname or "")
        assert bname == "Bambino"
        assert "Керуен" in (addr or "")

    def test_withdrawal(self):
        btype, bname, addr, city = parse_details(
            "Банкомат Iliyasa Zhansugurova 8k1", "Withdrawal"
        )
        assert btype is None
        assert bname == "Банкомат Iliyasa Zhansugurova 8k1"

    def test_blocked_suffix_stripped(self):
        btype, bname, addr, city = parse_details(
            "Magnum Cash&Carry - The amount is blocked. The bank expects the confirmation of the payment system.",
            "Purchases",
        )
        assert btype is None
        assert bname == "Magnum Cash&Carry"

    def test_blocked_suffix_russian(self):
        btype, bname, addr, city = parse_details(
            "Ресторан Trattoria Highvill - Сумма заблокирована. Банк ожидает подтверждения.",
            "Purchases",
        )
        assert btype is None
        assert bname == "Ресторан Trattoria Highvill"


# ---------------------------------------------------------------------------
#  2. City extraction — trailing city
# ---------------------------------------------------------------------------

class TestCityExtraction:
    def test_trailing_russian_city(self):
        btype, bname, addr, city = parse_details(
            "Рестопарк AVATARIYA Тараз", "Purchases"
        )
        assert city == "TARAZ"
        assert bname == "Рестопарк AVATARIYA"

    def test_trailing_english_city(self):
        btype, bname, addr, city = parse_details(
            "Burger King Mega Astana", "Purchases"
        )
        assert city == "ASTANA"

    def test_trailing_astana_russian(self):
        btype, bname, addr, city = parse_details(
            "12 месяцев Астана", "Purchases"
        )
        assert city == "ASTANA"
        assert bname == "месяцев"

    def test_konoha_astana(self):
        btype, bname, addr, city = parse_details(
            "Konoha astana", "Purchases"
        )
        assert city == "ASTANA"
        assert bname == "Konoha"

    def test_h202_taraz(self):
        btype, bname, addr, city = parse_details(
            "H-202 Тараз", "Purchases"
        )
        assert city == "TARAZ"
        assert bname == "H"

    def test_salon_taraz(self):
        btype, bname, addr, city = parse_details(
            "Все для салона Тараз", "Purchases"
        )
        assert city == "TARAZ"
        assert bname == "Все для салона"

    def test_boba_turkestan(self):
        btype, bname, addr, city = parse_details(
            "Боба Туркестан", "Purchases"
        )
        assert city == "TURKESTAN"
        assert bname == "Боба"

    def test_g_dot_prefix(self):
        """Russian 'г.' prefix means 'city of'."""
        btype, bname, addr, city = parse_details(
            "Z-1 г. Астана", "Purchases"
        )
        assert city == "ASTANA"
        assert bname == "Z"

    def test_g_dot_prefix_variant(self):
        btype, bname, addr, city = parse_details(
            "Z-313 г. Астана", "Purchases"
        )
        assert city == "ASTANA"
        assert bname == "Z"

    def test_almaty_trailing(self):
        btype, bname, addr, city = parse_details(
            "Музей Алматы", "Purchases"
        )
        assert city == "ALMATY"
        assert bname == "Музей"

    def test_nur_sultan_normalized(self):
        btype, bname, addr, city = parse_details(
            "Z - 300 Нур-Султан", "Purchases"
        )
        assert city == "ASTANA"

    def test_no_city(self):
        btype, bname, addr, city = parse_details(
            "Green Food", "Purchases"
        )
        assert city is None

    def test_central_astana(self):
        btype, bname, addr, city = parse_details(
            "Central Astana", "Purchases"
        )
        assert city == "ASTANA"
        assert bname == "Central"


# ---------------------------------------------------------------------------
#  3. Known places → address extraction
# ---------------------------------------------------------------------------

class TestKnownPlaces:
    def test_burger_king_mega_silk_way(self):
        btype, bname, addr, city = parse_details(
            "Burger King Mega Silk Way", "Purchases"
        )
        assert bname == "Burger King"
        assert "Mega Silk Way" in (addr or "")
        assert city is None

    def test_bahandi_mega_silk_way(self):
        btype, bname, addr, city = parse_details(
            "Bahandi Mega Silk Way", "Purchases"
        )
        assert bname == "Bahandi"
        assert "Mega Silk Way" in (addr or "")

    def test_burger_king_dostyk_plaza_almaty(self):
        btype, bname, addr, city = parse_details(
            "Burger King Dostyk Plaza Almaty", "Purchases"
        )
        assert bname == "Burger King"
        assert "Dostyk Plaza" in (addr or "")
        assert city == "ALMATY"

    def test_burger_king_khan_shatyr_astana(self):
        btype, bname, addr, city = parse_details(
            "Burger King Khan Shatyr Astana", "Purchases"
        )
        assert bname == "Burger King"
        assert "Khan Shatyr" in (addr or "")
        assert city == "ASTANA"

    def test_burger_king_esentai_mall_almaty(self):
        btype, bname, addr, city = parse_details(
            "Burger King Esentai Mall Almaty", "Purchases"
        )
        assert bname == "Burger King"
        assert "Esentai Mall" in (addr or "")
        assert city == "ALMATY"

    def test_burger_king_keruen_astana(self):
        btype, bname, addr, city = parse_details(
            "Burger King Keruen Astana", "Purchases"
        )
        assert bname == "Burger King"
        assert "Keruen" in (addr or "")
        assert city == "ASTANA"

    def test_nanduk_mega_silkway(self):
        btype, bname, addr, city = parse_details(
            "Nanduk mega silkway", "Purchases"
        )
        assert bname == "Nanduk"
        assert addr is not None
        assert "mega silkway" in addr.lower()

    def test_parking_trc_mega_silkway(self):
        btype, bname, addr, city = parse_details(
            "Парковка. ТРЦ Mega Silkway", "Purchases"
        )
        # "ТРЦ Mega Silkway" is a known place → goes into address
        assert addr is not None
        assert "ТРЦ Mega Silkway" in addr
        assert bname == "Парковка"

    def test_parking_trc_keruen(self):
        btype, bname, addr, city = parse_details(
            "Парковка. ТРЦ Keruen", "Purchases"
        )
        assert addr is not None
        assert "ТРЦ Keruen" in addr
        assert bname == "Парковка"

    def test_parking_trc_khan_shatyr_russian(self):
        btype, bname, addr, city = parse_details(
            "Парковка. ТРЦ Хан Шатыр", "Purchases"
        )
        assert addr is not None
        assert "ТРЦ Хан Шатыр" in addr
        assert bname == "Парковка"

    def test_medeo(self):
        btype, bname, addr, city = parse_details(
            "Медео. Аренда коньков", "Purchases"
        )
        assert bname == "Аренда коньков"
        assert addr == "Медео"

    def test_trc_mega_only(self):
        """ТРЦ Mega alone should keep both."""
        btype, bname, addr, city = parse_details(
            "ТРЦ Mega", "Purchases"
        )
        assert bname == "ТРЦ Mega"
        assert addr == "ТРЦ Mega"

    def test_tc_dostyk_plaza(self):
        btype, bname, addr, city = parse_details(
            "TC Dostyk Plaza", "Purchases"
        )
        assert bname == "TC Dostyk Plaza"
        assert "TC Dostyk Plaza" in (addr or "")

    def test_mega_supermarket_astana(self):
        btype, bname, addr, city = parse_details(
            "ТС Mega Supermarket Astana", "Purchases"
        )
        assert city == "ASTANA"
        assert bname == "Supermarket"
        assert addr == "ТС Mega"

    def test_keruen_cyrillic(self):
        btype, bname, addr, city = parse_details(
            "Bambino Керуен", "Purchases"
        )
        assert bname == "Bambino"
        assert "Керуен" in (addr or "")

    def test_nazarbayev_university(self):
        btype, bname, addr, city = parse_details(
            "Пабло Назарбаев Университет", "Purchases"
        )
        assert bname == "Пабло"
        assert "Назарбаев Университет" in (addr or "")

    def test_trc_eurasia(self):
        btype, bname, addr, city = parse_details(
            "ТРЦ Евразия", "Purchases"
        )
        # When the entire text is a known place, keep it as both
        assert bname == "ТРЦ Евразия"
        assert addr == "ТРЦ Евразия"

    def test_tc_mega_supermarket(self):
        """ТЦ Мега extracted as place, rest is business name."""
        btype, bname, addr, city = parse_details(
            "ТЦ Мега Супермаркет Золотое Яблоко", "Purchases"
        )
        assert bname == "Супермаркет Золотое Яблоко"
        assert addr == "ТЦ Мега"

    def test_mega_coffee_no_place_match(self):
        """Single-word 'MEGA' should NOT match when it's part of a brand name."""
        btype, bname, addr, city = parse_details(
            "ТОО Mega Coffee", "Purchases"
        )
        assert btype == "TOO"
        assert bname == "Mega Coffee"
        assert addr is None

    def test_place_only_keeps_both(self):
        """If extracting known place leaves bname empty, both should be set."""
        btype, bname, addr, city = parse_details(
            "Керуен", "Purchases"
        )
        assert bname == "Керуен"
        assert addr == "Керуен"

    def test_no_place(self):
        btype, bname, addr, city = parse_details(
            "Corner Meal", "Purchases"
        )
        assert addr is None


# ---------------------------------------------------------------------------
#  4. Foreign amounts & trailing numbers
# ---------------------------------------------------------------------------

class TestForeignAmountStripping:
    def test_trailing_store_code(self):
        btype, bname, addr, city = parse_details(
            "KFC 12780", "Purchases"
        )
        assert bname == "KFC"

    def test_digit_tokens_stripped(self):
        btype, bname, addr, city = parse_details(
            "АЗС 14", "Purchases"
        )
        assert bname == "АЗС"

    def test_underscore_to_space(self):
        btype, bname, addr, city = parse_details(
            "ECO_SHINE", "Purchases"
        )
        assert bname == "ECO SHINE"

    def test_asterisk_to_space(self):
        btype, bname, addr, city = parse_details(
            "GOOGLE *Google One", "Purchases"
        )
        assert bname == "GOOGLE Google One"

    def test_hash_stripped(self):
        btype, bname, addr, city = parse_details(
            "Zerde #201", "Purchases"
        )
        assert bname == "Zerde"

    def test_exclamation_stripped(self):
        btype, bname, addr, city = parse_details(
            "Рахмет!!! Спасибо!!!", "Purchases"
        )
        assert bname == "Рахмет Спасибо"

    def test_slash_to_space(self):
        btype, bname, addr, city = parse_details(
            "Meloman/Marwin", "Purchases"
        )
        assert bname == "Meloman Marwin"

    def test_foreign_amount_cny_stripped(self):
        btype, bname, addr, city = parse_details(
            "PINDUODUO (- 36,49 CNY)", "Purchases"
        )
        assert bname == "PINDUODUO"

    def test_foreign_amount_usd_stripped(self):
        btype, bname, addr, city = parse_details(
            "PAYPAL *COHEECREATI (- 6,97 USD)", "Purchases"
        )
        assert bname == "PAYPAL COHEECREATI"

    def test_foreign_amount_with_prefix(self):
        btype, bname, addr, city = parse_details(
            "FFT*PINDUODUO4 (- 89,77 CNY)", "Purchases"
        )
        assert bname == "FFT PINDUODUO4"

    def test_kz_suffix_stripped(self):
        btype, bname, addr, city = parse_details(
            "OKADZAKI.KZ", "Purchases"
        )
        assert bname == "OKADZAKI"

    def test_dn_market_hyphen_kept(self):
        btype, bname, addr, city = parse_details(
            "DN-маркет", "Purchases"
        )
        assert bname == "DN-маркет"


# ---------------------------------------------------------------------------
#  5. Non-purchase operations → all None (renumbered)
# ---------------------------------------------------------------------------

class TestNonPurchase:
    @pytest.mark.parametrize("op", [
        "Transfers", "Replenishment", "Others",
    ])
    def test_returns_none(self, op):
        btype, bname, addr, city = parse_details("ТОО Kaspi Travel", op)
        assert btype is None
        assert bname is None
        assert addr is None
        assert city is None


# ---------------------------------------------------------------------------
#  5. Deduplication by details column
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
#  6. normalize_csv produces correct columns
# ---------------------------------------------------------------------------

class TestNormalizeCsvOutput:
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
            df_out = normalize_csv(inp, out)

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
            df_out = normalize_csv(inp, out)

        assert df_out.iloc[0]["city"] == "TARAZ"
        assert df_out.iloc[0]["business_name"] == "Рестопарк AVATARIYA"

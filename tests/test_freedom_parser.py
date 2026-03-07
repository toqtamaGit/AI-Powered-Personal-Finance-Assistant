"""
Tests for parsers/freedom.py — bilingual operation detection.

Bug reproduced: the old OP_RE regex only contained Russian operation names
(Покупка, Платеж, …). English PDFs from Freedom Bank use English names
(Acquiring, Payment, …), so ~86% of transactions ended up with operation=None.

Run:  python -m pytest tests/test_freedom_parser.py -v
"""
import re
import pytest
from parsers.freedom import (
    OP_RE,
    OP_MAP,
    _OP_LOOKUP,
    normalize_operation,
    normalize_hyphens,
    extract_fields_from_chunk,
    try_parse_tables,
    map_table_headers,
    normalize_ws,
    fix_similar_details,
)
import pandas as pd


# ---------------------------------------------------------------------------
#  1. The root cause: OP_RE must match BOTH English and Russian operations
# ---------------------------------------------------------------------------

# Old regex that caused the bug (kept here to prove it fails on English)
OLD_OP_RE = re.compile(
    r"\b(Покупка|Платеж|Пополнение|Перевод|Снятие|Другое|Комиссия|Возврат|Кэшбэк)\b"
)

ENGLISH_OPS = [
    "Acquiring", "Payment", "Replenishment", "Transfer",
    "Withdrawal", "Others", "Commission", "Refund",
    "Cashback", "Pending amount",
]

RUSSIAN_OPS = [
    "Покупка", "Платеж", "Пополнение", "Перевод",
    "Снятие", "Другое", "Комиссия", "Возврат",
    "Кэшбэк", "Сумма в обработке",
]


class TestOldRegexFails:
    """Prove the old regex does NOT match English operation names."""

    @pytest.mark.parametrize("op", ENGLISH_OPS)
    def test_old_regex_misses_english(self, op):
        text = f"29.05.2025 15,992.00 KZT {op} TOO GLASMAN TRADE 3 ASTANA"
        assert OLD_OP_RE.search(text) is None, (
            f"Old regex should NOT match English '{op}' — that was the bug"
        )


class TestNewRegexMatches:
    """The fixed OP_RE must match all operations in both languages."""

    @pytest.mark.parametrize("op", ENGLISH_OPS)
    def test_matches_english(self, op):
        text = f"29.05.2025 15,992.00 KZT {op} SOME MERCHANT ASTANA"
        m = OP_RE.search(text)
        assert m is not None, f"OP_RE must match English '{op}'"
        assert m.group(1) == op

    @pytest.mark.parametrize("op", RUSSIAN_OPS)
    def test_matches_russian(self, op):
        text = f"29.05.2025 15 992,00 KZT {op} ТОО «Рога и копыта»"
        m = OP_RE.search(text)
        assert m is not None, f"OP_RE must match Russian '{op}'"
        assert m.group(1) == op


# ---------------------------------------------------------------------------
#  2. normalize_operation returns canonical English + correct sign
# ---------------------------------------------------------------------------

class TestNormalizeOperation:
    """All aliases must map to canonical English name with correct sign."""

    @pytest.mark.parametrize("raw,expected_canon,expected_sign", [
        ("Acquiring",        "Acquiring",      -1),
        ("Покупка",          "Acquiring",      -1),
        ("Replenishment",    "Replenishment",  +1),
        ("Пополнение",       "Replenishment",  +1),
        ("Refund",           "Refund",         +1),
        ("Возврат",          "Refund",         +1),
        ("Cashback",         "Cashback",       +1),
        ("Кэшбэк",          "Cashback",       +1),
        ("Transfer",         "Transfer",       -1),
        ("Перевод",          "Transfer",       -1),
        ("Pending amount",   "Pending amount", -1),
        ("Сумма в обработке","Pending amount", -1),
        ("Payment",          "Payment",        -1),
        ("Платеж",           "Payment",        -1),
    ])
    def test_canonical_and_sign(self, raw, expected_canon, expected_sign):
        canon, sign = normalize_operation(raw)
        assert canon == expected_canon
        assert sign == expected_sign

    def test_none_input(self):
        canon, sign = normalize_operation(None)
        assert canon is None
        assert sign == -1

    def test_unknown_operation_passthrough(self):
        canon, sign = normalize_operation("SomeUnknownOp")
        assert canon == "SomeUnknownOp"
        assert sign == -1


# ---------------------------------------------------------------------------
#  3. extract_fields_from_chunk — end-to-end on realistic text chunks
# ---------------------------------------------------------------------------

class TestExtractFieldsFromChunk:
    """Simulates real PDF text lines and checks operation extraction."""

    def test_english_acquiring(self):
        chunk = ["29.05.2025 15,992.00 KZT Acquiring TOO GLASMAN TRADE 3 ASTANA"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Acquiring"
        assert rec["date"] == "2025-05-29"
        assert rec["amount"] is not None and rec["amount"] < 0  # expense

    def test_english_replenishment(self):
        chunk = ["01.06.2025 250,000.00 KZT Replenishment Между своими счетами"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Replenishment"
        assert rec["amount"] is not None and rec["amount"] > 0  # income

    def test_english_transfer(self):
        chunk = ["15.06.2025 50,000.00 KZT Transfer Card-to-card"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Transfer"
        assert rec["amount"] is not None and rec["amount"] < 0

    def test_english_pending_amount(self):
        chunk = ["06.09.2025 140.00 KZT Pending amount TOO KIKIS MS"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Pending amount"
        assert rec["amount"] is not None and rec["amount"] < 0

    def test_russian_acquiring(self):
        chunk = ["29.05.2025 15 992,00 KZT Покупка ТОО «GLASMAN TRADE 3» ASTANA"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Acquiring"  # normalized to English
        assert rec["amount"] is not None and rec["amount"] < 0

    def test_russian_replenishment(self):
        chunk = ["01.06.2025 250 000,00 KZT Пополнение Между своими счетами"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Replenishment"
        assert rec["amount"] is not None and rec["amount"] > 0

    def test_russian_pending(self):
        chunk = ["06.09.2025 950,00 KZT Сумма в обработке HP CANTEEN NUR-SULTAN KZ"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Pending amount"


# ---------------------------------------------------------------------------
#  4. map_table_headers — English table headers
# ---------------------------------------------------------------------------

class TestMapTableHeaders:
    """Table header detection must work for both Russian and English PDFs."""

    def test_english_headers(self):
        df = pd.DataFrame([
            ["Date", "The amount", "Currency", "Operation", "Details"],
            ["29.05.2025", "15,992.00", "KZT", "Acquiring", "TOO GLASMAN"],
        ])
        result = map_table_headers(df)
        assert "date" in result.columns
        assert "operation" in result.columns
        assert "details" in result.columns
        assert len(result) == 1  # header row consumed

    def test_russian_headers(self):
        df = pd.DataFrame([
            ["Дата", "Сумма", "Валюта", "Операция", "Детали"],
            ["29.05.2025", "15 992,00", "KZT", "Покупка", "ТОО GLASMAN"],
        ])
        result = map_table_headers(df)
        assert "date" in result.columns
        assert "operation" in result.columns
        assert "details" in result.columns
        assert len(result) == 1

    def test_header_not_first_row(self):
        """Header may appear in row 2 (e.g. section title on row 0)."""
        df = pd.DataFrame([
            ["Pending amount. Waiting for confirmation", None, None, None, None],
            [None, None, None, None, None],
            ["Date", "The amount", "Currency", "Operation", "Details"],
            ["29.05.2025", "15,992.00", "KZT", "Acquiring", "TOO GLASMAN"],
        ])
        result = map_table_headers(df)
        assert "date" in result.columns
        assert len(result) == 1  # only the data row remains


# ---------------------------------------------------------------------------
#  5. fix_similar_details — correct PDF generation artifacts without dropping rows
# ---------------------------------------------------------------------------

class TestFixSimilarDetails:
    """Corrupted details should be replaced by the longer similar version."""

    def test_corrupted_detail_gets_fixed(self):
        """NUR-ULTAN (broken) should be replaced by NUR-SULTAN (full)."""
        records = [
            {"date": "2025-07-05", "amount": -120.0, "details": "SUPERMARKET GALMART NUR-SULTAN KZ", "merchant": None},
            {"date": "2025-07-06", "amount": -200.0, "details": "SUPERMARKET GALMART NUR-ULTAN KZ", "merchant": None},
        ]
        result = fix_similar_details(records)
        assert len(result) == 2, "No rows should be removed"
        assert result[1]["details"] == "SUPERMARKET GALMART NUR-SULTAN KZ"

    def test_all_rows_preserved(self):
        """Even exact duplicates are kept — only details text is corrected."""
        records = [
            {"date": "2025-09-23", "amount": -465.0, "details": "MAGNUM.ASF46.KCO-2. ASTANA KZ", "merchant": None},
            {"date": "2025-09-23", "amount": -465.0, "details": "MAGNUM.ASF46.KCO-2. ASTANA KZ", "merchant": None},
        ]
        result = fix_similar_details(records)
        assert len(result) == 2

    def test_different_details_not_touched(self):
        """Rows with <85% similarity must NOT be altered."""
        records = [
            {"date": "2025-08-02", "amount": -4250.0, "details": "SMART RESTAURANTS ALMATY KZ", "merchant": None},
            {"date": "2025-08-02", "amount": -4250.0, "details": "Плательщик:Арстангалиев Получатель:Филиал", "merchant": None},
        ]
        result = fix_similar_details(records)
        assert result[0]["details"] == "SMART RESTAURANTS ALMATY KZ"
        assert result[1]["details"] == "Плательщик:Арстангалиев Получатель:Филиал"

    def test_longest_wins(self):
        """Among similar details, the longest string becomes the canonical one."""
        records = [
            {"date": "2025-07-05", "amount": -120.0, "details": "MOMYSHULY STATION 1 ALMATY KZ", "merchant": None},
            {"date": "2025-07-06", "amount": -150.0, "details": "MOMYSHULY STATION 1 ALMA KZ", "merchant": None},
            {"date": "2025-07-07", "amount": -180.0, "details": "MOMYSHULY STATION 1 ALMATY KZ EXTRA", "merchant": None},
        ]
        result = fix_similar_details(records)
        # The longest version should propagate to the shorter similar ones
        assert result[0]["details"] == "MOMYSHULY STATION 1 ALMATY KZ EXTRA"
        assert result[1]["details"] == "MOMYSHULY STATION 1 ALMATY KZ EXTRA"
        assert result[2]["details"] == "MOMYSHULY STATION 1 ALMATY KZ EXTRA"

    def test_empty_records(self):
        assert fix_similar_details([]) == []

    def test_equal_length_similar_details_unified(self):
        """Equal-length details with >85% similarity should be unified."""
        records = [
            {"date": "2025-06-01", "amount": -500.0, "details": "IP BERSINOVA S.IM.B.MOMY H KZ", "merchant": "IP BERSINOVA"},
            {"date": "2025-06-01", "amount": -500.0, "details": "IP BERSINOVA S.IM.B.MOMY H KZ", "merchant": "IP BERSINOVA"},
            {"date": "2025-06-02", "amount": -300.0, "details": "IP BERSINOVA S.IM.B.MOMYSH KZ", "merchant": "IP BERSINOVA"},
        ]
        result = fix_similar_details(records)
        assert len(result) == 3
        # All should have the same canonical detail
        assert result[0]["details"] == result[1]["details"] == result[2]["details"]

    def test_equal_length_most_frequent_wins(self):
        """When equal length, the most frequent variant becomes canonical."""
        records = [
            {"date": "2025-06-01", "amount": -100.0, "details": "CAFE MOMYSH ASTANA KZ", "merchant": None},
            {"date": "2025-06-02", "amount": -100.0, "details": "CAFE MOMYSH ASTANA KZ", "merchant": None},
            {"date": "2025-06-03", "amount": -100.0, "details": "CAFE MOMYSH ASTANA KZ", "merchant": None},
            {"date": "2025-06-04", "amount": -100.0, "details": "CAFE MOMY H ASTANA KZ", "merchant": None},
        ]
        result = fix_similar_details(records)
        # The 3-occurrence version should win over the 1-occurrence version
        for r in result:
            assert r["details"] == "CAFE MOMYSH ASTANA KZ"


# ---------------------------------------------------------------------------
#  6. Replenishment, Others, Transfer operations should have merchant=None
# ---------------------------------------------------------------------------

class TestNoMerchantOperations:
    """Replenishment, Others, and Transfer rows must not extract a merchant."""

    def test_replenishment_no_merchant(self):
        chunk = ["02.06.2025 +90.28 KZT Replenishment Пополнение. Выплата процентов по вкладу"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Replenishment"
        assert rec["merchant"] is None

    def test_others_no_merchant(self):
        chunk = ["02.08.2025 4,250.00 KZT Others Плательщик:Арстангалиев Рауан"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Others"
        assert rec["merchant"] is None

    def test_transfer_no_merchant(self):
        chunk = ["15.06.2025 50,000.00 KZT Transfer Перевод с карты на карту"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Transfer"
        assert rec["merchant"] is None

    def test_acquiring_still_has_merchant(self):
        chunk = ["29.05.2025 15,992.00 KZT Acquiring TOO GLASMAN TRADE 3 ASTANA"]
        rec = extract_fields_from_chunk(chunk)
        assert rec["operation"] == "Acquiring"
        assert rec["merchant"] is not None


# ---------------------------------------------------------------------------
#  7. Hyphen normalization — no spaces around dashes in details
# ---------------------------------------------------------------------------

class TestNormalizeHyphens:
    """Spaces before/after dashes must be removed in details."""

    def test_space_after_dash(self):
        assert normalize_hyphens("NUR- SULTAN") == "NUR-SULTAN"

    def test_space_before_dash(self):
        assert normalize_hyphens("NUR -SULTAN") == "NUR-SULTAN"

    def test_spaces_around_dash(self):
        assert normalize_hyphens("NUR - SULTAN") == "NUR-SULTAN"

    def test_no_space_unchanged(self):
        assert normalize_hyphens("NUR-SULTAN") == "NUR-SULTAN"

    def test_en_dash_normalized(self):
        assert normalize_hyphens("NUR\u2013 SULTAN") == "NUR-SULTAN"

    def test_em_dash_normalized(self):
        assert normalize_hyphens("NUR\u2014SULTAN") == "NUR-SULTAN"

    def test_extract_fields_no_space_around_dash(self):
        chunk = ["29.05.2025 1,538.00 KZT Acquiring MAGNUM CASH CARRY NF46 NUR- SULTAN"]
        rec = extract_fields_from_chunk(chunk)
        assert "NUR-SULTAN" in rec["details"]
        assert "NUR- " not in rec["details"]
        assert " -SULTAN" not in rec["details"]

    def test_tangiers_lounge(self):
        chunk = ["23.07.2025 41,899.00 KZT Acquiring TANGIERS LOUNGE EC ASTANA Nur- Sultan"]
        rec = extract_fields_from_chunk(chunk)
        assert "Nur-Sultan" in rec["details"]
        assert "Nur- " not in rec["details"]


# ---------------------------------------------------------------------------
# 10. Quote/bracket stripping in parser
# ---------------------------------------------------------------------------

class TestQuoteStripping:
    """Quotes and brackets should be stripped from details at parse time."""

    def test_double_quotes_stripped_chunk(self):
        chunk = ['23.07.2025 1,000.00 KZT Acquiring TOO "UNIQLOAST" ASTANA KZ']
        rec = extract_fields_from_chunk(chunk)
        assert '"' not in rec["details"]
        assert "UNIQLOAST" in rec["details"]

    def test_single_quotes_stripped_chunk(self):
        chunk = ["23.07.2025 500.00 KZT Acquiring TOO 'SHOP' ASTANA KZ"]
        rec = extract_fields_from_chunk(chunk)
        assert "'" not in rec["details"]
        assert "SHOP" in rec["details"]

    def test_brackets_stripped_chunk(self):
        chunk = ["23.07.2025 200.00 KZT Acquiring NAME [EXTRA] (INFO) ASTANA KZ"]
        rec = extract_fields_from_chunk(chunk)
        assert "[" not in rec["details"]
        assert "]" not in rec["details"]
        assert "(" not in rec["details"]
        assert ")" not in rec["details"]

    def test_double_double_quotes_chunk(self):
        chunk = ['23.07.2025 300.00 KZT Acquiring TOO ""KIKIS MS"" ASTANA KZ']
        rec = extract_fields_from_chunk(chunk)
        assert '"' not in rec["details"]
        assert "KIKIS" in rec["details"]

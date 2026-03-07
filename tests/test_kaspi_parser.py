"""
Tests for parsers/kaspi.py — bilingual operation detection & data quality.

Run:  python -m pytest tests/test_kaspi_parser.py -v
"""
import re
import pytest
from parsers.kaspi import (
    OP_RE,
    OP_MAP,
    _OP_LOOKUP,
    normalize_operation,
    parse_operation_line,
    is_header_like,
    normalize_ws,
    NO_MERCHANT_OPS,
)


# ---------------------------------------------------------------------------
#  1. OP_RE matches both English and Russian operation names
# ---------------------------------------------------------------------------

ENGLISH_OPS = ["Purchases", "Transfers", "Replenishment", "Commission",
               "Refund", "Cashback", "Withdrawal", "Others"]

RUSSIAN_OPS = ["Покупка", "Перевод", "Пополнение", "Комиссия",
               "Возврат", "Кэшбэк", "Снятие", "Разное"]


class TestOpRegex:
    @pytest.mark.parametrize("op", ENGLISH_OPS)
    def test_matches_english(self, op):
        text = f"some text {op} more text"
        m = OP_RE.search(text)
        assert m is not None, f"OP_RE must match English '{op}'"

    @pytest.mark.parametrize("op", RUSSIAN_OPS)
    def test_matches_russian(self, op):
        text = f"some text {op} more text"
        m = OP_RE.search(text)
        assert m is not None, f"OP_RE must match Russian '{op}'"


# ---------------------------------------------------------------------------
#  2. normalize_operation returns canonical name + correct sign
# ---------------------------------------------------------------------------

class TestNormalizeOperation:
    @pytest.mark.parametrize("raw,expected_canon,expected_sign", [
        ("Purchases",     "Purchases",     -1),
        ("Покупка",       "Purchases",     -1),
        ("Transfers",     "Transfers",     -1),
        ("Перевод",       "Transfers",     -1),
        ("Replenishment", "Replenishment", +1),
        ("Пополнение",    "Replenishment", +1),
        ("Commission",    "Commission",    -1),
        ("Комиссия",      "Commission",    -1),
        ("Refund",        "Refund",        +1),
        ("Возврат",       "Refund",        +1),
        ("Cashback",      "Cashback",      +1),
        ("Кэшбэк",       "Cashback",      +1),
        ("Withdrawal",    "Withdrawal",    -1),
        ("Снятие",        "Withdrawal",    -1),
        ("Others",        "Others",        -1),
        ("Разное",        "Others",        -1),
    ])
    def test_canonical_and_sign(self, raw, expected_canon, expected_sign):
        canon, sign = normalize_operation(raw)
        assert canon == expected_canon
        assert sign == expected_sign

    def test_none_input(self):
        canon, sign = normalize_operation(None)
        assert canon is None
        assert sign == -1

    def test_unknown_passthrough(self):
        canon, sign = normalize_operation("UnknownOp")
        assert canon == "UnknownOp"
        assert sign == -1


# ---------------------------------------------------------------------------
#  3. parse_operation_line — end-to-end line parsing
# ---------------------------------------------------------------------------

class TestParseOperationLine:
    def test_english_purchase(self):
        line = "03.11.25 - 500,00 ₸ Purchases TOO \"KASPI MAGAZIN\""
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["operation"] == "Purchases"
        assert rec["date"] == "2025-11-03"
        assert rec["amount"] < 0

    def test_english_replenishment(self):
        line = "12.11.25 + 20 000,00 ₸ Replenishment Сандугаш К."
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["operation"] == "Replenishment"
        assert rec["amount"] > 0
        assert rec["merchant"] is None  # no merchant for Replenishment

    def test_english_transfer(self):
        line = "12.11.25 - 10 000,00 ₸ Transfers Кербез К."
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["operation"] == "Transfers"
        assert rec["amount"] < 0
        assert rec["merchant"] is None  # no merchant for Transfers

    def test_russian_purchase(self):
        line = "13.11.25 - 110,00 ₸ Покупка Avtobys. Оплата"
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["operation"] == "Purchases"  # normalized
        assert rec["amount"] < 0

    def test_russian_replenishment(self):
        line = "12.11.25 + 50 000,00 ₸ Пополнение Зарплата"
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["operation"] == "Replenishment"
        assert rec["amount"] > 0
        assert rec["merchant"] is None

    def test_russian_transfer(self):
        line = "12.11.25 - 5 000,00 ₸ Перевод Иван А."
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["operation"] == "Transfers"
        assert rec["amount"] < 0
        assert rec["merchant"] is None

    def test_no_date_returns_none(self):
        assert parse_operation_line("some random text") is None

    def test_header_line_returns_none(self):
        assert parse_operation_line("Date Amount Transaction Details") is None


# ---------------------------------------------------------------------------
#  4. No merchant for Replenishment, Transfers, Others
# ---------------------------------------------------------------------------

class TestNoMerchantOperations:
    @pytest.mark.parametrize("op_en,op_ru", [
        ("Replenishment", "Пополнение"),
        ("Transfers", "Перевод"),
        ("Others", "Разное"),
    ])
    def test_no_merchant_english(self, op_en, op_ru):
        line = f"01.11.25 - 1 000,00 ₸ {op_en} Some details here"
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["merchant"] is None

    def test_purchase_has_merchant(self):
        line = "01.11.25 - 1 000,00 ₸ Purchases MAGNUM store"
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["merchant"] is not None


# ---------------------------------------------------------------------------
#  5. is_header_like — filters junk lines
# ---------------------------------------------------------------------------

class TestHeaderFilter:
    @pytest.mark.parametrize("line", [
        "ВЫПИСКА",
        "Kaspi Gold balance statement for the period from 13.11.24 to 13.11.25",
        "Дата Сумма Операция Детали",
        "Date Amount Transaction Details",
        "Краткое содержание операций по карте:",
        "Transaction summary: Cash withdrawal limits with no commission:",
        "JSC «Kaspi Bank», BIC CASPKZKA, www.kaspi.kz",
        "Доступно на 13.11.25 + 15 545,48 ₸",
        "Card balance 13.11.25: + 444 947,74 ₸",
        "Валюта счета: тенге",
        "Currency: KZT",
    ])
    def test_header_detected(self, line):
        assert is_header_like(line), f"Should be filtered: {line}"

    @pytest.mark.parametrize("line", [
        "13.11.25 - 110,00 ₸ Покупка Avtobys",
        "12.11.25 + 20 000,00 ₸ Replenishment Сандугаш К.",
        "TOO KASPI MAGAZIN",
    ])
    def test_transaction_not_filtered(self, line):
        assert not is_header_like(line), f"Should NOT be filtered: {line}"


# ---------------------------------------------------------------------------
#  6. Quote/bracket stripping in parser
# ---------------------------------------------------------------------------

class TestQuoteStripping:
    """Quotes and brackets should be stripped from details at parse time."""

    def test_double_quotes_stripped(self):
        line = '13.11.25 - 1 000,00 ₸ Purchases TOO "UNIQLOAST"'
        rec = parse_operation_line(line)
        assert rec is not None
        assert '"' not in rec["details"]
        assert "UNIQLOAST" in rec["details"]

    def test_single_quotes_stripped(self):
        line = "13.11.25 - 500,00 ₸ Purchases TOO 'SHOP'"
        rec = parse_operation_line(line)
        assert rec is not None
        assert "'" not in rec["details"]
        assert "SHOP" in rec["details"]

    def test_brackets_stripped(self):
        line = "13.11.25 - 200,00 ₸ Purchases NAME [EXTRA] (INFO)"
        rec = parse_operation_line(line)
        assert rec is not None
        assert "[" not in rec["details"]
        assert "]" not in rec["details"]
        assert "(" not in rec["details"]
        assert ")" not in rec["details"]

    def test_no_quotes_unchanged(self):
        line = "13.11.25 - 300,00 ₸ Purchases PLAIN MERCHANT"
        rec = parse_operation_line(line)
        assert rec is not None
        assert rec["details"] == "PLAIN MERCHANT"

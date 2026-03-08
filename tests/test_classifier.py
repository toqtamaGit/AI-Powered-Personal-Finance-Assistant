"""Tests for the transaction classifier."""

import pytest

from llm.classifier import classify_transaction, classify_transactions, _keyword_classify


# ---------------------------------------------------------------------------
# Keyword fallback classifier
# ---------------------------------------------------------------------------

class TestKeywordClassify:
    def test_transfer(self):
        assert _keyword_classify("Перевод на карту") == "переводы"

    def test_transport(self):
        assert _keyword_classify("YANDEX.GO ALMATY") == "транспорт"

    def test_restaurant(self):
        assert _keyword_classify("Coffee House Cafe") == "рестораны и кафе"

    def test_supermarket(self):
        assert _keyword_classify("Magnum Cash&Carry") == "супермаркеты"

    def test_subscription(self):
        assert _keyword_classify("SPOTIFY PREMIUM") == "подписки"

    def test_marketplace(self):
        assert _keyword_classify("Wildberries заказ") == "маркетплейсы"

    def test_default(self):
        assert _keyword_classify("Random shop XYZ") == "покупки"


# ---------------------------------------------------------------------------
# classify_transaction (uses model if available, else keyword fallback)
# ---------------------------------------------------------------------------

class TestClassifyTransaction:
    def test_returns_string(self):
        result = classify_transaction("MAGNUM ASTANA")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_details(self):
        assert classify_transaction("") == "другое"

    def test_none_details(self):
        assert classify_transaction(None) == "другое"

    def test_whitespace_only(self):
        assert classify_transaction("   ") == "другое"


# ---------------------------------------------------------------------------
# classify_transactions (batch)
# ---------------------------------------------------------------------------

class TestClassifyTransactions:
    def test_adds_category_field(self):
        txs = [
            {"details": "MAGNUM ASTANA"},
            {"details": "YANDEX.GO"},
            {"details": "Перевод"},
        ]
        result = classify_transactions(txs)
        assert len(result) == 3
        for tx in result:
            assert "category" in tx
            assert isinstance(tx["category"], str)

    def test_preserves_existing_fields(self):
        txs = [{"details": "Coffee", "amount": -500, "date": "2024-01-01"}]
        result = classify_transactions(txs)
        assert result[0]["amount"] == -500
        assert result[0]["date"] == "2024-01-01"

    def test_empty_list(self):
        result = classify_transactions([])
        assert result == []

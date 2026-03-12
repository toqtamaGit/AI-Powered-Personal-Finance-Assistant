"""Contract tests for the Flask server endpoints."""

import io
import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from server.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Flask test client (uses the module-level app which includes /health)."""
    from server.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_fake_pdf():
    """Return a BytesIO that mimics a PDF upload (content doesn't matter when mocked)."""
    return (io.BytesIO(b"%PDF-1.4 fake"), "statement.pdf")


def _sample_df():
    """Return a small DataFrame matching pipeline output shape."""
    return pd.DataFrame([
        {
            "date": "2024-11-13",
            "amount_raw": "- 1 500,00 KZT",
            "amount": -1500.0,
            "currency": "KZT",
            "operation": "Purchases",
            "details": "YANDEX GO ALMATY",
            "bank": "Kaspi Bank",
            "source_file": "statement.pdf",
            "business_type": None,
            "business_name": "YANDEX GO",
            "address": None,
            "city": "ALMATY",
            "oked_code": "56101",
            "oked_description": "Restaurants",
            "category": "транспорт",
        },
        {
            "date": "2024-11-14",
            "amount_raw": "- 5 000,00 KZT",
            "amount": -5000.0,
            "currency": "KZT",
            "operation": "Purchases",
            "details": "MAGNUM ASTANA",
            "bank": "Kaspi Bank",
            "source_file": "statement.pdf",
            "business_type": None,
            "business_name": "MAGNUM",
            "address": None,
            "city": "ASTANA",
            "oked_code": "47111",
            "oked_description": "Supermarkets",
            "category": "супермаркеты",
        },
    ])


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /detect-bank
# ---------------------------------------------------------------------------

class TestDetectBank:
    def test_no_file(self, client):
        resp = client.post("/detect-bank")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_filename(self, client):
        resp = client.post(
            "/detect-bank",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_non_pdf(self, client):
        resp = client.post(
            "/detect-bank",
            data={"file": (io.BytesIO(b"hello"), "data.csv")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "PDF" in resp.get_json()["error"]

    @patch("server.routes.bank.extract_text_from_pdf", return_value="Kaspi Bank statement")
    @patch("server.routes.bank.detect_bank_from_text", return_value="Kaspi Bank")
    def test_detect_kaspi(self, mock_detect, mock_extract, client):
        resp = client.post(
            "/detect-bank",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["bank"] == "Kaspi Bank"
        assert data["detected"] is True

    @patch("server.routes.bank.extract_text_from_pdf", return_value="Unknown bank text")
    @patch("server.routes.bank.detect_bank_from_text", return_value=None)
    def test_detect_unknown(self, mock_detect, mock_extract, client):
        resp = client.post(
            "/detect-bank",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["bank"] is None
        assert data["detected"] is False


# ---------------------------------------------------------------------------
# /classify — response contract
# ---------------------------------------------------------------------------

class TestClassify:
    def test_no_file(self, client):
        resp = client.post("/classify")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_non_pdf(self, client):
        resp = client.post(
            "/classify",
            data={"file": (io.BytesIO(b"hello"), "data.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    @patch("server.routes.transactions.process_pdf", return_value=_sample_df())
    def test_response_shape(self, mock_process, client):
        """Verify the JSON contract: bank, transactions[], summary."""
        resp = client.post(
            "/classify",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()

        # Top-level keys
        assert "bank" in data
        assert "transactions" in data
        assert "summary" in data
        assert isinstance(data["transactions"], list)
        assert isinstance(data["summary"], dict)

        # Bank name
        assert data["bank"] == "Kaspi Bank"

        # Summary shape
        assert "total_transactions" in data["summary"]
        assert "by_category" in data["summary"]
        assert data["summary"]["total_transactions"] == 2

    @patch("server.routes.transactions.process_pdf", return_value=_sample_df())
    def test_transaction_fields(self, mock_process, client):
        """Each transaction must contain the enriched fields."""
        resp = client.post(
            "/classify",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        tx = data["transactions"][0]

        required_fields = [
            "date", "amount", "currency", "operation", "details",
            "bank", "source_file", "category",
        ]
        for field in required_fields:
            assert field in tx, f"Missing field: {field}"

        # Enriched fields (may be None but must be present)
        enriched_fields = [
            "business_type", "business_name", "address", "city",
            "oked_code", "oked_description",
        ]
        for field in enriched_fields:
            assert field in tx, f"Missing enriched field: {field}"

    @patch("server.routes.transactions.process_pdf", return_value=_sample_df())
    def test_category_summary(self, mock_process, client):
        """Category summary should aggregate correctly."""
        resp = client.post(
            "/classify",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        by_cat = data["summary"]["by_category"]

        assert by_cat["транспорт"] == 1
        assert by_cat["супермаркеты"] == 1

    @patch("server.routes.transactions.process_pdf")
    def test_skip_oked_param(self, mock_process, client):
        """Query param skip_oked=1 should be forwarded to process_pdf."""
        mock_process.return_value = _sample_df()
        client.post(
            "/classify?skip_oked=1",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        _, kwargs = mock_process.call_args
        assert kwargs.get("skip_oked") is True

    @patch("server.routes.transactions.process_pdf")
    def test_skip_classify_param(self, mock_process, client):
        """Query param skip_classify=1 should be forwarded to process_pdf."""
        mock_process.return_value = _sample_df()
        client.post(
            "/classify?skip_classify=1",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        _, kwargs = mock_process.call_args
        assert kwargs.get("skip_classify") is True

    @patch("server.routes.transactions.process_pdf", side_effect=ValueError("No parser"))
    def test_value_error_returns_400(self, mock_process, client):
        resp = client.post(
            "/classify",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert data["transactions"] == []

    @patch("server.routes.transactions.process_pdf", side_effect=RuntimeError("boom"))
    def test_unexpected_error_returns_500(self, mock_process, client):
        resp = client.post(
            "/classify",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data
        assert data["transactions"] == []


# ---------------------------------------------------------------------------
# /classify — no NaN leak
# ---------------------------------------------------------------------------

class TestNaNSerialization:
    @patch("server.routes.transactions.process_pdf")
    def test_nan_replaced_with_null(self, mock_process, client):
        """NaN values must be serialized as JSON null, not 'NaN' string."""
        df = _sample_df()
        df.loc[0, "business_type"] = None
        df.loc[0, "address"] = None
        mock_process.return_value = df

        resp = client.post(
            "/classify",
            data={"file": _make_fake_pdf()},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        tx = data["transactions"][0]
        assert tx["business_type"] is None
        assert tx["address"] is None
        # Ensure no "NaN" string leaked
        raw = resp.get_data(as_text=True)
        assert "NaN" not in raw

# AI-Powered Personal Finance Assistant

A machine learning-powered finance assistant that parses bank statement PDFs, extracts transactions, normalizes business details, fetches OKED codes, and classifies transactions by category.

## Installation

```bash
git clone https://github.com/yourusername/AI-Powered-Personal-Finance-Assistant.git
cd AI-Powered-Personal-Finance-Assistant

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Server

Start the REST API:

```bash
python -m server.app
```

The server runs at `http://localhost:5001`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/detect-bank` | Detect bank from a PDF |
| `POST` | `/classify` | Full pipeline: PDF → enriched JSON |

### `/detect-bank`

Upload a PDF, get the detected bank name.

```bash
curl -X POST -F "file=@statement.pdf" http://localhost:5001/detect-bank
```

```json
{ "bank": "Kaspi Bank", "detected": true }
```

### `/classify`

Upload a PDF, get parsed, normalized, OKED-enriched, and classified transactions.

```bash
curl -X POST -F "file=@statement.pdf" http://localhost:5001/classify

# Skip OKED fetching (faster, no network)
curl -X POST -F "file=@statement.pdf" "http://localhost:5001/classify?skip_oked=1"

# Skip ML classification
curl -X POST -F "file=@statement.pdf" "http://localhost:5001/classify?skip_classify=1"
```

```json
{
  "bank": "Kaspi Bank",
  "transactions": [
    {
      "date": "2024-11-13",
      "amount": -1500.0,
      "currency": "KZT",
      "operation": "Purchases",
      "details": "YANDEX GO ALMATY",
      "business_name": "YANDEX GO",
      "city": "ALMATY",
      "oked_description": "Taxi services",
      "category": "транспорт"
    }
  ],
  "summary": {
    "total_transactions": 50,
    "by_category": { "транспорт": 10, "супермаркеты": 15 }
  }
}
```

## CLI Pipeline

Process PDFs directly from the command line:

```bash
# Single PDF
python categorize.py --input statement.pdf --out result.csv

# Folder of PDFs
python categorize.py --input data/pdfs/ --out data/final/all.csv

# Skip OKED (faster, no network)
python categorize.py --input statement.pdf --out result.csv --skip-oked

# Skip ML classification
python categorize.py --input statement.pdf --out result.csv --skip-classify

# Control OKED fetch threads (default: 5)
python categorize.py --input data/pdfs/ --out result.csv --workers 10
```

The pipeline auto-detects the bank and runs every step:

```
PDF → Detect Bank → Parse → Normalize → Fetch OKED → ML Classify → CSV
```

### Output Columns

| Column | Description |
|--------|-------------|
| `date` | Transaction date (YYYY-MM-DD) |
| `amount_raw` | Original amount string from PDF |
| `amount` | Parsed numeric amount (signed) |
| `currency` | Currency code (KZT) |
| `operation` | Operation type (Purchases, Transfers, etc.) |
| `details` | Raw transaction details from PDF |
| `bank` | Detected bank name |
| `source_file` | Source PDF filename |
| `business_type` | Entity type (IP, TOO, LLC, etc.) |
| `business_name` | Cleaned business/merchant name |
| `address` | Known place or mall (if detected) |
| `city` | City name (normalized to English) |
| `oked_code` | OKED classification code from kompra.kz |
| `oked_description` | OKED activity description |
| `category` | ML-predicted spending category |

## Manual Steps

You can also run individual pipeline stages separately.

### Parse a PDF

```python
from parsers.kaspi import KaspiParser

parser = KaspiParser()
transactions, rejects = parser.parse("statement.pdf")
```

### Normalize details

```python
from parsers.normalize_kaspi import KaspiNormalizer

normalizer = KaspiNormalizer()
df = normalizer.normalize_csv("parsed.csv", "normalized.csv")
```

### Fetch OKED codes

```python
from llm.fetch_oked import process_dataframe

df = process_dataframe(df, workers=5)
```

### Classify transactions

```python
from llm.classifier import classify_transaction

category = classify_transaction("MAGNUM ASTANA")
```

### Train the ML model

```python
from llm.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train("data/bank_statements_labeled.csv")
trainer.save_model("llm/models/")
```

## Testing

```bash
# Unit tests only (fast, no I/O)
pytest tests/unit/

# Integration tests (reads files, no network)
pytest tests/integration/

# Everything except e2e
pytest tests/ -m "not e2e"

# Full suite including e2e (requires network)
pytest tests/
```

## Supported Banks

| Bank | Parser | Normalizer |
|------|--------|------------|
| Kaspi Bank | `parsers/kaspi.py` | `parsers/normalize_kaspi.py` |
| Freedom Bank Kazakhstan | `parsers/freedom.py` | `parsers/normalize_freedom.py` |

## Project Structure

```
├── categorize.py              # Main pipeline (PDF → enriched CSV)
├── parsers/
│   ├── __init__.py            # REGISTRY: bank → (parser, normalizer)
│   ├── base.py                # BaseParser / BaseNormalizer
│   ├── constants.py           # Shared constants (cities, places, regexes)
│   ├── kaspi.py               # Kaspi Bank PDF parser
│   ├── freedom.py             # Freedom Bank PDF parser
│   ├── normalize_kaspi.py     # Kaspi normalizer
│   └── normalize_freedom.py   # Freedom normalizer
├── llm/
│   ├── classifier.py          # ML classification (LinearSVC)
│   ├── model_trainer.py       # Model training
│   ├── fetch_oked.py          # OKED fetching from kompra.kz
│   ├── bank_detector.py       # Bank detection
│   ├── pdf_extractor.py       # PDF text extraction
│   └── models/                # Trained model artifacts
├── server/
│   ├── app.py                 # Flask app factory
│   └── routes/                # API endpoints
├── tests/
│   ├── unit/                  # Pure function tests, no I/O
│   ├── integration/           # File I/O, real PDFs, no network
│   └── e2e/                   # Full pipeline with network
└── data/
    ├── pdfs/                  # Source bank statement PDFs
    └── final/                 # Pipeline output CSVs
```

## Tech Stack

- **Python 3**, Flask, flask-cors
- **ML**: scikit-learn (LinearSVC, TfidfVectorizer)
- **PDF**: pdfplumber, PyPDF2
- **Data**: pandas, NumPy
- **OKED**: kompra.kz API

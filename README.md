# AI-Powered Personal Finance Assistant

A machine learning-powered finance assistant that parses bank statement PDFs, extracts transactions, normalizes business details, fetches OKED codes, and classifies transactions by category — all in a single command.

## Quick Start

```bash
# Process a single PDF — produces a fully enriched CSV
python categorize.py --input statement.pdf --out result.csv

# Process a folder of PDFs
python categorize.py --input data/pdfs/ --out data/final/all.csv

# Skip OKED fetching (faster, no network needed)
python categorize.py --input statement.pdf --out result.csv --skip-oked

# Skip ML classification
python categorize.py --input statement.pdf --out result.csv --skip-classify

# Control OKED fetch parallelism (default: 5 threads)
python categorize.py --input data/pdfs/ --out result.csv --workers 10
```

The pipeline auto-detects the bank from the PDF content and runs every step automatically:

```
PDF → Detect Bank → Parse Transactions → Normalize Details → Fetch OKED → ML Classify → CSV
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
| `business_type` | Extracted entity type (IP, TOO, LLC, etc.) |
| `business_name` | Cleaned business/merchant name |
| `address` | Known place or mall (if detected) |
| `city` | Extracted city name (normalized to English) |
| `oked_code` | OKED classification code from kompra.kz |
| `oked_description` | OKED activity description |
| `category` | ML-predicted spending category |

## Features

- **Single-Command Pipeline**: `categorize.py` chains parsing, normalization, OKED lookup, and classification
- **Auto Bank Detection**: Identifies Kaspi Bank and Freedom Bank Kazakhstan from PDF text
- **Business Detail Extraction**: Extracts business type, name, city, and address from transaction details
- **OKED Integration**: Fetches business activity codes from kompra.kz API (multithreaded)
- **ML Classification**: Categorizes transactions using a trained LinearSVC model
- **Robust PDF Parsing**: Dual extraction (pdfplumber + PyPDF2 fallback), handles Cyrillic text

## Project Structure

```
├── categorize.py           # Main pipeline script (PDF → enriched CSV)
│
├── parsers/                # Bank-specific parsers & normalizers
│   ├── __init__.py         # REGISTRY: maps bank name → (parser, normalizer)
│   ├── base.py             # BaseParser / BaseNormalizer abstract classes
│   ├── constants.py        # Shared constants (cities, places, regexes)
│   ├── kaspi.py            # Kaspi Bank PDF parser
│   ├── freedom.py          # Freedom Bank PDF parser
│   ├── normalize_kaspi.py  # Kaspi detail normalizer
│   └── normalize_freedom.py # Freedom detail normalizer
│
├── llm/                    # ML & API components
│   ├── classifier.py       # Transaction classification (LinearSVC)
│   ├── model_trainer.py    # Model training script
│   ├── fetch_oked.py       # OKED code fetching from kompra.kz
│   ├── bank_detector.py    # Bank detection from PDF text
│   ├── pdf_extractor.py    # PDF text extraction utility
│   └── models/             # Trained ML model artifacts
│
├── server/                 # Flask REST API
│   ├── app.py              # App factory
│   └── routes/             # API endpoints
│
├── tests/                  # pytest test suite
├── data/                   # PDFs, CSVs, and training data
│   ├── pdfs/               # Source bank statement PDFs
│   ├── final/              # Pipeline output CSVs (fully enriched)
│   └── bank_statements_labeled.csv  # Labeled training data
└── notebooks/              # Jupyter notebooks for experimentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Powered-Personal-Finance-Assistant.git
   cd AI-Powered-Personal-Finance-Assistant
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask flask-cors scikit-learn pandas numpy pdfplumber PyPDF2 openpyxl requests
   ```

4. **Run the pipeline**
   ```bash
   python categorize.py --input data/pdfs/ --out data/final/all.csv
   ```

5. **Or run the server**
   ```bash
   python -m server.app
   ```
   The server will start at `http://localhost:5001`

## Pipeline Steps

### 1. Bank Detection
Scans PDF text for bank-specific keywords (e.g. "Kaspi", "Freedom Finance") and routes to the correct parser.

### 2. PDF Parsing
Bank-specific parsers extract structured transaction rows from PDFs. Handles table layouts, multi-line entries, Cyrillic text, and OCR artifacts.

### 3. Normalization
Extracts structured fields from raw transaction details:
- **Business type** — IP, TOO, LLP, LLC
- **Business name** — cleaned name with punctuation, digits, and domain suffixes removed
- **Address** — known places (malls: Mega, Khan Shatyr, Dostyk Plaza, etc.)
- **City** — Kazakhstan city names (Russian and English, including genitive forms)

### 4. OKED Fetching
Queries kompra.kz API to find the business activity code for each unique business name. Uses multithreaded requests with BLEU-score matching. Online services (Spotify, Coursera, Netflix, etc.) are tagged directly without API calls.

### 5. ML Classification
Applies a pre-trained LinearSVC model to assign a spending category to each transaction based on its details text.

## Supported Banks

| Bank | Parser | Normalizer | Status |
|------|--------|------------|--------|
| Kaspi Bank | `parsers/kaspi.py` | `parsers/normalize_kaspi.py` | Full support |
| Freedom Bank Kazakhstan | `parsers/freedom.py` | `parsers/normalize_freedom.py` | Full support |

## Training the Model

```python
from llm.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train('data/bank_statements_labeled.csv')
trainer.save_model('llm/models/')
```

## Tech Stack

- **Python 3**, Flask
- **ML**: scikit-learn (LinearSVC, TfidfVectorizer)
- **PDF**: pdfplumber, PyPDF2
- **Data**: Pandas, NumPy
- **OKED**: kompra.kz API

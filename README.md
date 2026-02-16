# AI-Powered Personal Finance Assistant

A machine learning-powered finance assistant that parses bank statement PDFs, extracts transactions, and automatically classifies them by category. Built with Flask and scikit-learn, designed for integration with mobile applications.

## Features

- **Multi-Bank PDF Parsing**: Supports Kaspi Bank, Freedom Bank Kazakhstan, Halyk Bank, and Jusan Bank statement formats
- **ML-Based Classification**: Automatically categorizes transactions using a trained LinearSVC model with TF-IDF vectorization
- **REST API**: Flask-based API with CORS support for mobile app integration
- **Robust Text Extraction**: Dual PDF extraction (pdfplumber + PyPDF2 fallback)
- **Text Normalization**: Handles Cyrillic text, OCR errors, and fuzzy merchant matching

## Project Structure

```
├── server/                 # Flask REST API
│   ├── app.py             # App factory & initialization
│   └── routes/            # API endpoints
│       ├── bank.py        # Bank detection endpoint
│       └── transactions.py # Transaction parsing & classification
│
├── llm/                   # ML & NLP components
│   ├── classifier.py      # Transaction classification (LinearSVC)
│   ├── model_trainer.py   # Model training pipeline
│   ├── bank_detector.py   # Bank detection from PDF text
│   ├── pdf_extractor.py   # PDF text extraction
│   ├── transaction_parser.py # Parsing coordinator
│   ├── normalize.py       # Text normalization utilities
│   └── models/            # Trained ML models
│
├── parsers/               # Bank-specific PDF parsers
│   ├── kaspi.py          # Kaspi Bank parser
│   └── freedom.py        # Freedom Bank parser
│
├── notebooks/             # Jupyter notebooks for experimentation
├── data/                  # Training data & datasets
└── statements/            # Sample bank statements (gitignored)
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

4. **Run the server**
   ```bash
   python -m server.app
   ```
   The server will start at `http://localhost:5001`

## API Endpoints

### Health Check
```
GET /health
```
Returns: `{"status": "ok"}`

### Detect Bank
```
POST /detect-bank
Content-Type: multipart/form-data

file: <PDF file>
```
Returns:
```json
{
  "bank": "Kaspi Bank",
  "detected": true
}
```

### Parse & Classify Transactions
```
POST /classify
Content-Type: multipart/form-data

file: <PDF file>
```
Returns:
```json
{
  "bank": "Kaspi Bank",
  "transactions": [
    {
      "date": "2024-11-13",
      "amount": -1500.0,
      "currency": "KZT",
      "operation": "Покупка",
      "merchant": "YANDEX.GO",
      "details": "YANDEX.GO ALMATY KZ",
      "category": "транспорт"
    }
  ],
  "summary": {
    "total_transactions": 50,
    "by_category": {
      "транспорт": 10,
      "рестораны и кафе": 8,
      "супермаркеты": 12
    }
  }
}
```

## Transaction Categories

The classifier assigns transactions to the following categories:

| Category | Description |
|----------|-------------|
| `переводы` | Transfers |
| `покупки` | General purchases |
| `транспорт` | Transport/Taxi |
| `рестораны и кафе` | Restaurants & Cafes |
| `супермаркеты` | Supermarkets |
| `подписки` | Subscriptions |
| `маркетплейсы` | Marketplaces |
| `другое` | Other |

## Training the Model

To retrain the classification model with new data:

```python
from llm.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train('data/bank_statements_labeled.csv')
trainer.save_model('llm/models/')
```

Or use the Jupyter notebook at `notebooks/classification.ipynb` for interactive training and experimentation.

## Supported Banks

| Bank | Parser | Status |
|------|--------|--------|
| Kaspi Bank | `parsers/kaspi.py` | Full support |
| Freedom Bank Kazakhstan | `parsers/freedom.py` | Full support |
| Halyk Bank | Freedom parser fallback | Partial support |
| Jusan Bank | Freedom parser fallback | Partial support |

## Tech Stack

- **Backend**: Python 3.14, Flask 3.1.2
- **ML/NLP**: scikit-learn (LinearSVC, TfidfVectorizer)
- **PDF Processing**: pdfplumber, PyPDF2
- **Data Processing**: Pandas, NumPy

## License

MIT License

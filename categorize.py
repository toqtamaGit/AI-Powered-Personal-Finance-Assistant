#!/usr/bin/env python3
"""Full categorization pipeline: PDF → parse → normalize → OKED → classify → CSV.

Usage:
    python categorize.py --input statement.pdf --out result.csv
    python categorize.py --input pdf_folder/ --out result.csv --workers 5
    python categorize.py --input statement.pdf --out result.csv --skip-oked
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.bank_detector import detect_bank_from_text
from llm.pdf_extractor import extract_text_from_pdf
from llm.classifier import classify_transaction
from llm.fetch_oked import process_dataframe as fetch_oked_dataframe
from parsers import REGISTRY


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def collect_pdfs(input_path: str) -> List[str]:
    """Collect PDF file paths from a file or directory (recursive)."""
    if os.path.isdir(input_path):
        files: List[str] = []
        for root, _, names in os.walk(input_path):
            for name in names:
                if name.lower().endswith('.pdf'):
                    files.append(os.path.join(root, name))
        files.sort()
        if not files:
            raise SystemExit(f"No PDF files found in {input_path}")
        return files

    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        return [input_path]

    raise SystemExit(f"Not a valid PDF file or directory: {input_path}")


def detect_bank(pdf_path: str) -> str:
    """Detect the bank from a PDF file's text content."""
    text = extract_text_from_pdf(pdf_path)
    bank = detect_bank_from_text(text)
    if not bank:
        raise ValueError(f"Could not detect bank from {os.path.basename(pdf_path)}")
    return bank


# ---------------------------------------------------------------------------
# Single-PDF processing (used by both CLI and server)
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: str,
    workers: int = 5,
    skip_oked: bool = False,
    skip_classify: bool = False,
) -> pd.DataFrame:
    """Process a single PDF through the full pipeline and return a DataFrame.

    Raises:
        ValueError: if bank cannot be detected or no parser is registered.
    """
    basename = os.path.basename(pdf_path)
    bank_name = detect_bank(pdf_path)

    if bank_name not in REGISTRY:
        raise ValueError(f"No parser registered for '{bank_name}'")

    parser_cls, norm_cls = REGISTRY[bank_name]

    # Parse
    transactions, _ = parser_cls().parse(pdf_path)
    if not transactions:
        raise ValueError(f"No transactions extracted from {basename}")

    for t in transactions:
        t.setdefault('bank', bank_name)
        t.setdefault('source_file', basename)

    df = pd.DataFrame(transactions)

    # Canonical column order
    base_cols = [
        'date', 'amount_raw', 'amount', 'currency',
        'operation', 'details', 'bank', 'source_file',
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    df = df[base_cols + [c for c in df.columns if c not in base_cols]]

    # Normalize
    df = norm_cls().normalize_dataframe(df)

    # OKED
    if not skip_oked:
        df = fetch_oked_dataframe(df, workers=workers)

    # Classify
    if not skip_classify:
        df['category'] = df['details'].apply(classify_transaction)

    # Sort by date
    if 'date' in df.columns:
        df['_dt'] = df['date'].apply(
            lambda x: datetime.strptime(str(x), "%Y-%m-%d") if pd.notna(x) else pd.NaT
        )
        df = df.sort_values('_dt', kind='stable').drop(columns=['_dt'])

    return df


# ---------------------------------------------------------------------------
# Batch pipeline (multiple PDFs → single CSV)
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_path: str,
    workers: int = 5,
    skip_oked: bool = False,
    skip_classify: bool = False,
) -> pd.DataFrame:
    """Run the full categorization pipeline.

    Steps:
        1. Collect PDFs
        2. Detect bank per PDF
        3. Parse transactions (via registered parser)
        4. Normalize details (via registered normalizer)
        5. Fetch OKED codes (optional)
        6. ML classify (optional)
        7. Save final CSV
    """
    pdfs = collect_pdfs(input_path)
    print(f"Found {len(pdfs)} PDF(s)\n")

    all_rows: List[Dict] = []
    log: List[str] = []

    for pdf_path in pdfs:
        basename = os.path.basename(pdf_path)
        try:
            # Step 1: detect bank
            bank_name = detect_bank(pdf_path)
            print(f"[{basename}] Detected: {bank_name}")

            # Step 2: look up parser + normalizer
            if bank_name not in REGISTRY:
                log.append(f"[SKIP] {basename}: No parser registered for {bank_name}")
                continue

            parser_cls, _ = REGISTRY[bank_name]
            parser = parser_cls()

            # Step 3: parse
            transactions, rejects = parser.parse(pdf_path)
            if rejects:
                log.append(f"[INFO] {basename}: {len(rejects)} rejected lines")

            for t in transactions:
                t.setdefault('bank', bank_name)
                t.setdefault('source_file', basename)
            all_rows.extend(transactions)

            print(f"  Parsed {len(transactions)} transactions")

        except Exception as e:
            log.append(f"[ERROR] {basename}: {e}")
            print(f"[{basename}] ERROR: {e}")

    if not all_rows:
        raise SystemExit("No transactions extracted from any PDF.")

    df = pd.DataFrame(all_rows)

    # Ensure canonical column order
    base_cols = [
        'date', 'amount_raw', 'amount', 'currency',
        'operation', 'details', 'bank', 'source_file',
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    df = df[base_cols + [c for c in df.columns if c not in base_cols]]

    # Step 4: normalize per bank
    print("\n--- Normalizing ---")
    normalized_parts: List[pd.DataFrame] = []
    for bank_name, bank_df in df.groupby('bank', sort=False):
        if bank_name in REGISTRY:
            _, norm_cls = REGISTRY[bank_name]
            normalizer = norm_cls()
            print(f"\n[{bank_name}]")
            bank_df = normalizer.normalize_dataframe(bank_df.copy())
        normalized_parts.append(bank_df)

    df = pd.concat(normalized_parts, ignore_index=True)

    # Step 5: fetch OKED
    if not skip_oked:
        print("\n--- Fetching OKED codes ---")
        df = fetch_oked_dataframe(df, workers=workers)
    else:
        print("\n--- Skipping OKED (--skip-oked) ---")

    # Step 6: ML classify
    if not skip_classify:
        print("\n--- Classifying transactions ---")
        df['category'] = df['details'].apply(classify_transaction)
        print(f"  Classified {len(df)} transactions")
    else:
        print("\n--- Skipping classification (--skip-classify) ---")

    # Sort by date
    def _to_dt(x):
        try:
            return datetime.strptime(str(x), "%Y-%m-%d")
        except Exception:
            return pd.NaT

    if 'date' in df.columns:
        df['_dt'] = df['date'].apply(_to_dt)
        df = df.sort_values(by=['_dt', 'source_file'], kind='stable').drop(columns=['_dt'])

    # Step 7: save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nDone: {output_path} ({len(df)} transactions)")

    # Write log
    if log:
        log_path = os.path.splitext(output_path)[0] + '_pipeline_log.txt'
        with open(log_path, 'w', encoding='utf-8') as f:
            for line in log:
                f.write(line + '\n')
        print(f"Log: {log_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Full categorization pipeline: PDF → CSV with business info, OKED, and category."
    )
    ap.add_argument('--input', required=True, help="PDF file or folder of PDFs.")
    ap.add_argument('--out', required=True, help="Output CSV path.")
    ap.add_argument('--workers', type=int, default=5, help="OKED fetch threads (default: 5).")
    ap.add_argument('--skip-oked', action='store_true', help="Skip OKED code fetching.")
    ap.add_argument('--skip-classify', action='store_true', help="Skip ML classification.")
    args = ap.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.out,
        workers=args.workers,
        skip_oked=args.skip_oked,
        skip_classify=args.skip_classify,
    )


if __name__ == '__main__':
    main()

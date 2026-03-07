"""Normalize Kaspi Bank CSV details: deduplicate + extract business_type, business_name."""

import argparse
import os
import re
from typing import Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUSINESS_TYPE_RE = re.compile(
    r'\b(ИП|ТОО|TOO|IP|IPP|LLP|LLC)\b\.?', flags=re.IGNORECASE
)

# Operations where we extract business info
_PURCHASE_OPS = {'Purchases', 'Withdrawal'}


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------

def parse_details(
    details: str, operation: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extract business_type and business_name from Kaspi details.

    Only processes Purchases / Withdrawal operations.
    Returns (business_type, business_name).
    """
    if operation not in _PURCHASE_OPS:
        return None, None

    if not details or (isinstance(details, float) and pd.isna(details)):
        return None, None

    text = str(details).strip()

    # Strip trailing "- The amount is blocked..." / "- Сумма заблокирована..."
    text = re.sub(
        r'\s*-\s*(The amount is blocked|Сумма заблокирована).*$',
        '', text, flags=re.IGNORECASE,
    )

    # Extract business type (anywhere in text)
    _type_map = {'ИП': 'IP', 'ТОО': 'TOO'}
    m = BUSINESS_TYPE_RE.search(text)
    if m:
        business_type = m.group(1).upper()
        business_type = _type_map.get(business_type, business_type)
        business_name = (text[:m.start()] + ' ' + text[m.end():]).strip()
    else:
        business_type = None
        business_name = text

    # Clean up quotes/brackets that may remain in Kaspi merchant names
    business_name = re.sub(r'["\'\[\]\(\)]', '', business_name).strip()

    if not business_name:
        business_name = None

    return business_type, business_name


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def normalize_csv(input_file: str, output_file: str) -> pd.DataFrame:
    """Read Kaspi CSV, deduplicate by details, extract business columns, save."""
    df = pd.read_csv(input_file)

    # Deduplicate by details column (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=['details'], keep='first').reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Deduplicated: dropped {dropped} rows with duplicate details")

    # Parse each row
    parsed = df.apply(
        lambda row: parse_details(
            row.get('details', ''),
            row.get('operation', ''),
        ),
        axis=1,
    )

    df['business_type'] = [x[0] for x in parsed]
    df['business_name'] = [x[1] for x in parsed]

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    purch = df[df['operation'].isin(_PURCHASE_OPS)]
    print(f"Done: {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Purchases/Withdrawal: {len(purch)}")
    print(f"  business_type filled: {purch['business_type'].notna().sum()}")
    print(f"  business_name filled: {purch['business_name'].notna().sum()}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Normalize Kaspi CSV: deduplicate + extract business_type, business_name"
    )
    ap.add_argument("--input", required=True, help="Input CSV path.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    normalize_csv(args.input, args.out)


if __name__ == '__main__':
    main()

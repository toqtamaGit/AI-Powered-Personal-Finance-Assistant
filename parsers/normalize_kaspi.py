"""Normalize Kaspi Bank CSV details: deduplicate + extract business_type, business_name, address, city.

This module provides both a class-based API (``KaspiNormalizer``) and
backward-compatible module-level functions (``parse_details``, ``normalize_csv``).
"""

import argparse
import os
import re
from typing import Optional, Tuple

import pandas as pd

from parsers.base import BaseNormalizer
from parsers.constants import (
    BUSINESS_TYPE_RE,
    CITIES_EN,
    CITIES_RU_MAP,
    CITY_NORMALIZE,
    GENERIC_PLACES,
    KNOWN_PLACES,
    TYPE_MAP,
    normalize_city,
)

# ---------------------------------------------------------------------------
# Kaspi-specific constants
# ---------------------------------------------------------------------------

# Operations where we extract business info
_PURCHASE_OPS = {'Purchases', 'Withdrawal'}

# All known city tokens (for regex building)
_ALL_CITY_TOKENS = sorted(
    list(CITIES_EN) + list(CITIES_RU_MAP.keys()),
    key=len, reverse=True,
)

# Trailing city regex: city as last word(s), optionally preceded by "г."
_TRAILING_CITY_RE = re.compile(
    r'(?:\s+г\.)?'
    r'\s+(' + '|'.join(re.escape(c) for c in _ALL_CITY_TOKENS) + r')\s*$',
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core parsing (kept as module-level function for backward compat)
# ---------------------------------------------------------------------------

def parse_details(
    details: str, operation: str
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract business_type, business_name, address, city from Kaspi details.

    Only processes Purchases / Withdrawal operations.
    Returns (business_type, business_name, address, city).
    """
    if operation not in _PURCHASE_OPS:
        return None, None, None, None

    if not details or (isinstance(details, float) and pd.isna(details)):
        return None, None, None, None

    text = str(details).strip()

    # Strip trailing "- The amount is blocked..." / "- Сумма заблокирована..."
    text = re.sub(
        r'\s*-\s*(The amount is blocked|Сумма заблокирована).*$',
        '', text, flags=re.IGNORECASE,
    )

    # 1. Extract trailing city (optionally preceded by "г.")
    city = None
    m_city = _TRAILING_CITY_RE.search(text)
    if m_city:
        city = normalize_city(m_city.group(1))
        text = text[:m_city.start()].strip()

    # 2. Extract business type
    business_type = None
    m = BUSINESS_TYPE_RE.search(text)
    if m:
        business_type = m.group(1).upper()
        business_type = TYPE_MAP.get(business_type, business_type)
        text = (text[:m.start()] + ' ' + text[m.end():]).strip()

    # 3. Strip foreign currency amounts e.g. "(- 36,49 CNY)" before punct strip
    text = re.sub(r'\([-–]\s*[\d\s,.]+[A-Z]{3}\)', '', text).strip()

    # 4. Clean up quotes/brackets/punctuation → replace with spaces
    text = re.sub(r'["\'\[\]\(\)«»,;.*_#!/]', ' ', text)

    # 5. Hyphen before digit → space (e.g. CAFE-2 → CAFE 2, KASSA-1 → KASSA 1)
    text = re.sub(r'-(\d)', r' \1', text)

    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Known places → split into address
    address = None
    words = text.split()
    if words:
        text_upper = ' '.join(words).upper()
        for place in KNOWN_PLACES:
            if place in text_upper:
                place_word_count = len(place.split())
                # Find where the place starts in the word list
                matched = False
                for i in range(len(words) - place_word_count + 1):
                    segment = ' '.join(words[i:i + place_word_count]).upper()
                    if segment == place:
                        # Skip generic single-word places (MEGA/МЕГА) at
                        # position 0 when there are trailing words, to avoid
                        # false matches like "Mega Coffee".
                        if (i == 0 and place_word_count == 1
                                and len(words) > 1
                                and place in GENERIC_PLACES):
                            continue
                        addr_words = words[i:i + place_word_count]
                        name_words = words[:i] + words[i + place_word_count:]
                        address = ' '.join(addr_words).strip() or None
                        text = ' '.join(name_words).strip()
                        matched = True
                        break
                if matched:
                    break

    business_name = text.strip() if text.strip() else None

    # Strip purely-digit tokens (store / kassa / phone numbers)
    if business_name:
        bwords = [w for w in business_name.split() if not w.isdigit()]
        business_name = ' '.join(bwords).strip() or None

    # Strip trailing "KZ" domain suffix (from e.g. OZON.KZ → OZON KZ)
    if business_name and business_name.upper().endswith(' KZ'):
        business_name = business_name[:-3].strip() or None

    # If extracting the known place left business_name empty,
    # keep the place as both business_name and address
    if business_name is None and address is not None:
        business_name = address

    return business_type, business_name, address, city


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------

class KaspiNormalizer(BaseNormalizer):
    """Normalizer for Kaspi Bank transaction details."""

    @property
    def bank_name(self) -> str:
        return "Kaspi Bank"

    @property
    def business_ops(self) -> set:
        return _PURCHASE_OPS

    def parse_details(self, details: str, operation: str) -> tuple:
        return parse_details(details, operation)

    def print_summary(self, df: pd.DataFrame, output_file: str) -> None:
        purch = df[df['operation'].isin(_PURCHASE_OPS)]
        print(f"Done: {output_file}")
        print(f"  Total rows: {len(df)}")
        print(f"  Purchases/Withdrawal: {len(purch)}")
        print(f"  business_type filled: {purch['business_type'].notna().sum()}")
        print(f"  business_name filled: {purch['business_name'].notna().sum()}")
        print(f"  address filled: {purch['address'].notna().sum()}")
        print(f"  city filled: {purch['city'].notna().sum()}")


# Singleton instance
_normalizer = KaspiNormalizer()


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions
# ---------------------------------------------------------------------------

def normalize_csv(input_file: str, output_file: str) -> pd.DataFrame:
    """Read Kaspi CSV, deduplicate by details, extract business columns, save."""
    return _normalizer.normalize_csv(input_file, output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Normalize Kaspi CSV: deduplicate + extract business_type, business_name, address, city"
    )
    ap.add_argument("--input", required=True, help="Input CSV path.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    normalize_csv(args.input, args.out)


if __name__ == '__main__':
    main()

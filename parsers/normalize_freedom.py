"""Normalize Freedom Bank CSV details: deduplicate + extract business_type, business_name, address, city, country.

This module provides both a class-based API (``FreedomNormalizer``) and
backward-compatible module-level functions (``parse_freedom_details``,
``normalize_hyphens``, ``normalize_csv``, ``extract_city_from_text``).
"""

import argparse
import os
import re
from typing import List, Optional, Tuple

import pandas as pd

from parsers.base import BaseNormalizer
from parsers.constants import (
    BUSINESS_TYPE_RE,
    CITIES_EN,
    CITY_NORMALIZE,
    KNOWN_PLACES,
    TYPE_MAP,
)

# ---------------------------------------------------------------------------
# Freedom-specific constants
# ---------------------------------------------------------------------------

# Operations where we extract business info
_BUSINESS_OPS = {'Acquiring', 'Payment', 'Pending amount', 'Withdrawal'}

# Known Kazakhstan cities (upper-case for matching)
_KNOWN_CITIES = CITIES_EN

# Tokens to discard (district abbreviations etc.)
_DISCARD_TOKENS = {'Q.', 'Q'}

# Country code pattern: exactly 2 uppercase letters at the end
_COUNTRY_RE = re.compile(r'\s+([A-Z]{2})$')

# City pattern: known city optionally followed by "Q."
_CITY_RE = re.compile(
    r'\s+(' + '|'.join(
        re.escape(c) for c in sorted(_KNOWN_CITIES, key=len, reverse=True)
    ) + r')(?:\s+Q\.)?$',
    flags=re.IGNORECASE,
)

_TYPE_MAP = TYPE_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_hyphens(text: str) -> str:
    """Remove spaces around dashes, normalize en/em dash to ASCII hyphen."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text
    s = str(text)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s*-\s*", "-", s)
    return s


def _normalize_city_name(name: str) -> str:
    """Normalize city name (e.g. NUR-SULTAN -> ASTANA)."""
    upper = name.upper()
    return CITY_NORMALIZE.get(upper, upper)


def extract_city_from_text(text: str) -> Optional[str]:
    """Extract and normalize city name from text. Returns None if no city found."""
    if not text:
        return None
    upper = text.upper()
    for city in sorted(_KNOWN_CITIES, key=len, reverse=True):
        if city in upper:
            return _normalize_city_name(city)
    return None


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------

def parse_freedom_details(
    details: str, operation: str
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract business_type, business_name, address, city, country from Freedom details.

    Only processes Acquiring / Payment / Pending amount / Withdrawal operations.
    Returns (business_type, business_name, address, city, country).
    """
    if operation not in _BUSINESS_OPS:
        return None, None, None, None, None

    if not details or (isinstance(details, float) and pd.isna(details)):
        return None, None, None, None, None

    text = str(details).strip()

    # 1. Extract country code (trailing 2-letter uppercase)
    country = None
    m_country = _COUNTRY_RE.search(text)
    if m_country:
        country = m_country.group(1)
        text = text[:m_country.start()].strip()

    # 2. Extract city (trailing known city name, optionally with "Q." suffix)
    city = None
    m_city = _CITY_RE.search(text)
    if m_city:
        city = _normalize_city_name(m_city.group(1).upper())
        text = text[:m_city.start()].strip()

    # 3. Also check for city anywhere in the remaining text (dual-city patterns
    #    like "ASTANA Nur-Sultan" — the second city was stripped, detect the first)
    if city is None:
        city = extract_city_from_text(text)

    # 4. Extract business type
    business_type = None
    m_type = BUSINESS_TYPE_RE.search(text)
    if m_type:
        business_type = m_type.group(1).upper()
        business_type = _TYPE_MAP.get(business_type, business_type)
        text = (text[:m_type.start()] + ' ' + text[m_type.end():]).strip()

    # 5. Strip foreign currency amounts e.g. "(- 36,49 CNY)" before punct strip
    text = re.sub(r'\([-–]\s*[\d\s,.]+[A-Z]{3}\)', '', text).strip()

    # 6. Clean up quotes/brackets/punctuation → replace with spaces
    text = re.sub(r'["\'\[\]\(\)«».,;*_#!/]', ' ', text)

    # 6. Hyphen before digit → space (e.g. KASSA-1 → KASSA 1)
    text = re.sub(r'-(\d)', r' \1', text)

    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Strip city-alias tokens from the remaining text (NUR-SULTAN already
    #    captured — don't let it bleed into business name)
    words = text.split()
    words = [w for w in words if w.upper() not in CITY_NORMALIZE]

    # 7. Strip discard tokens (Q., Q)
    words = [w for w in words if w.upper().rstrip('.') not in
             {t.upper().rstrip('.') for t in _DISCARD_TOKENS}]

    if not words:
        return business_type, None, None, city, country

    # 8. Split into name_words / address_words using known places
    name_words = list(words)
    address_words: list[str] = []

    name_text_upper = ' '.join(name_words).upper()
    for place in KNOWN_PLACES:
        if place in name_text_upper:
            # Find where the place starts in the word list
            for i in range(len(name_words)):
                candidate = ' '.join(name_words[i:]).upper()
                if candidate.startswith(place):
                    address_words = name_words[i:] + address_words
                    name_words = name_words[:i]
                    break
            break

    business_name = ' '.join(name_words).strip() if name_words else None
    address = ' '.join(address_words).strip() if address_words else None

    # Strip purely-digit tokens (store / kassa / phone numbers)
    if business_name:
        bwords = [w for w in business_name.split() if not w.isdigit()]
        business_name = ' '.join(bwords).strip() or None

    # Strip trailing "KZ" domain suffix (from e.g. OZON.KZ → OZON KZ)
    if business_name and business_name.upper().endswith(' KZ'):
        business_name = business_name[:-3].strip() or None

    return business_type, business_name, address, city, country


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------

class FreedomNormalizer(BaseNormalizer):
    """Normalizer for Freedom Bank Kazakhstan transaction details."""

    @property
    def bank_name(self) -> str:
        return "Freedom Bank Kazakhstan"

    @property
    def business_ops(self) -> set:
        return _BUSINESS_OPS

    @property
    def detail_fields(self) -> List[str]:
        return ['business_type', 'business_name', 'address', 'city', 'country']

    def parse_details(self, details: str, operation: str) -> tuple:
        return parse_freedom_details(details, operation)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize hyphens in details before parsing."""
        if 'details' in df.columns:
            df['details'] = df['details'].apply(normalize_hyphens)
        return df

    def print_summary(self, df: pd.DataFrame, output_file: str) -> None:
        biz = df[df['operation'].isin(_BUSINESS_OPS)]
        print(f"Done: {output_file}")
        print(f"  Total rows: {len(df)}")
        print(f"  Business ops: {len(biz)}")
        print(f"  business_type filled: {biz['business_type'].notna().sum()}")
        print(f"  business_name filled: {biz['business_name'].notna().sum()}")
        print(f"  address filled: {biz['address'].notna().sum()}")
        print(f"  city filled: {biz['city'].notna().sum()}")
        print(f"  country filled: {biz['country'].notna().sum()}")


# Singleton instance
_normalizer = FreedomNormalizer()


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions
# ---------------------------------------------------------------------------

def normalize_csv(input_file: str, output_file: str) -> pd.DataFrame:
    """Read Freedom CSV, deduplicate by details, extract business columns, save."""
    return _normalizer.normalize_csv(input_file, output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Normalize Freedom CSV: deduplicate + extract business_type, business_name, address, city, country"
    )
    ap.add_argument("--input", required=True, help="Input CSV path.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    normalize_csv(args.input, args.out)


if __name__ == '__main__':
    main()

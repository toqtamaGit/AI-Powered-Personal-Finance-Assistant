"""Normalize Freedom Bank CSV details: extract business_type, business_name, address, city."""

import argparse
import os
import re
from typing import Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAZAKHSTAN_CITIES = [
    'ASTANA', 'ALMATY', 'SHYMKENT', 'KARAGANDA', 'TARAZ', 'PAVLODAR',
    'SEMEY', 'KOSTANAY', 'UST-KAMENOGORSK', 'PETROPAVLOVSK', 'ATYRAU',
    'ORAL', 'URALSK', 'KYZYLORDA', 'TALDYKORGAN', 'ARKALYK', 'AKTOBE',
    'EKIBASTUZ', 'RUDNY', 'ZHEZKAZGAN', 'TURKESTAN',
    'NUR-SULTAN',  # old name for Astana
]

# NUR-SULTAN variants all map to ASTANA
_CITY_NORMALIZE = {
    'NUR-SULTAN': 'ASTANA',
}

KNOWN_PLACES = [
    'MEGA SILKWAY', 'MEGA SILK WAY', 'DOSTYK PLAZA', 'AZIYA PARK',
    'MEGA', 'DOSTYK',
]

# Tokens to discard from address (district abbreviations etc.)
_DISCARD_TOKENS = {'Q.', 'Q'}

BUSINESS_TYPE_RE = re.compile(
    r'\b(IP|TOO|ТОО|IPP|LLP|LLC)\b\.?', flags=re.IGNORECASE
)


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


def _is_city(word: str) -> bool:
    """Check if a word is a known city (case-insensitive)."""
    return word.upper() in (c.upper() for c in KAZAKHSTAN_CITIES)


def _normalize_city_name(name: str) -> str:
    """Normalize city name (e.g. NUR-SULTAN -> ASTANA)."""
    upper = name.upper()
    return _CITY_NORMALIZE.get(upper, upper)


def extract_city_from_text(text: str) -> Optional[str]:
    """Extract and normalize city name from text. Returns None if no city found."""
    if not text:
        return None
    upper = text.upper()
    for city in KAZAKHSTAN_CITIES:
        if city in upper:
            return _normalize_city_name(city)
    return None


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------

def parse_details(
    details: str, operation: str
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parse the details column to extract business_type, business_name, address, city.

    Only processes Acquiring / Pending amount operations.
    Returns (business_type, business_name, address, city).
    """
    business_type = None
    business_name = None
    address = None
    city = None

    if operation not in ('Acquiring', 'Pending amount'):
        return business_type, business_name, address, city

    if not details or pd.isna(details):
        return business_type, business_name, address, city

    details = str(details)

    # --- Step 1: extract business type ---
    type_match = BUSINESS_TYPE_RE.search(details)
    if type_match:
        business_type = type_match.group(1).upper()
        details_no_type = details[:type_match.start()] + details[type_match.end():]
        details_no_type = details_no_type.strip()
    else:
        details_no_type = details

    # --- Step 2: clean whitespace (quotes already stripped by parser) ---
    cleaned = re.sub(r'\s+', ' ', details_no_type).strip()

    # --- Step 3: detect city early (before stripping tokens) ---
    city = extract_city_from_text(cleaned)

    # --- Step 4: strip city-alias tokens (NUR-SULTAN, Nur-Sultan) ---
    words = cleaned.split()
    words = [w for w in words if w.upper() not in _CITY_NORMALIZE]

    # --- Step 5: strip discard tokens (Q.) ---
    words = [w for w in words if w.upper().rstrip('.') not in
             {t.upper().rstrip('.') for t in _DISCARD_TOKENS}]

    if not words:
        return business_type, business_name, address, city

    # --- Step 6: split into name_words / address_words ---
    # Find position of KZ and last city word to determine split point
    kz_idx = None
    last_city_idx = None
    for i, w in enumerate(words):
        if w.upper() == 'KZ':
            kz_idx = i
        if _is_city(w):
            last_city_idx = i

    if last_city_idx is not None:
        # address starts at city
        split = last_city_idx
    elif kz_idx is not None:
        # no city word left (was stripped) — just KZ
        split = kz_idx
    else:
        # no KZ, no city — everything is business name
        split = len(words)

    name_words = words[:split]
    address_words = words[split:]

    # --- Step 8: known places move from name to address ---
    if name_words:
        name_text = ' '.join(name_words).upper()
        for place in KNOWN_PLACES:
            if name_text.endswith(place):
                place_word_count = len(place.split())
                if len(name_words) >= place_word_count:
                    address_words = name_words[-place_word_count:] + address_words
                    name_words = name_words[:-place_word_count]
                    break
            elif place in name_text:
                for i in range(len(name_words)):
                    if ' '.join(name_words[i:]).upper().startswith(place):
                        address_words = name_words[i:] + address_words
                        name_words = name_words[:i]
                        break
                break

    # --- Step 9: assemble outputs ---
    business_name = ' '.join(name_words).strip() if name_words else None
    address = ' '.join(address_words).strip() if address_words else None

    return business_type, business_name, address, city


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def normalize_csv(input_file: str, output_file: str) -> pd.DataFrame:
    """Read bank CSV, extract business columns, save to output_file."""
    df = pd.read_csv(input_file)

    # Normalize hyphens in details
    if 'details' in df.columns:
        df['details'] = df['details'].apply(normalize_hyphens)

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
    df['address'] = [x[2] for x in parsed]
    df['city'] = [x[3] for x in parsed]

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    acq = df[df['operation'].isin(['Acquiring', 'Pending amount'])]
    print(f"Done: {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Acquiring/Pending: {len(acq)}")
    print(f"  business_type filled: {acq['business_type'].notna().sum()}")
    print(f"  city filled: {acq['city'].notna().sum()}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Normalize Freedom CSV: extract business_type, business_name, address, city"
    )
    ap.add_argument("--input", required=True, help="Input CSV path.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    normalize_csv(args.input, args.out)


if __name__ == '__main__':
    main()

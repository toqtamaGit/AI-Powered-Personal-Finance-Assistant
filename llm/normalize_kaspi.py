"""Normalize Kaspi Bank CSV details: deduplicate + extract business_type, business_name, address, city."""

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

# Known Kazakhstan cities — English (upper-case canonical forms)
_CITIES_EN = {
    'ASTANA', 'ALMATY', 'SHYMKENT', 'KARAGANDA', 'TARAZ', 'PAVLODAR',
    'SEMEY', 'KOSTANAY', 'UST-KAMENOGORSK', 'PETROPAVLOVSK', 'ATYRAU',
    'ORAL', 'URALSK', 'KYZYLORDA', 'TALDYKORGAN', 'ARKALYK', 'AKTOBE',
    'EKIBASTUZ', 'RUDNY', 'ZHEZKAZGAN', 'TURKESTAN',
    'NUR-SULTAN',
}

# Russian city names → canonical English upper-case
_CITIES_RU_MAP = {
    'АСТАНА': 'ASTANA',
    'АСТАНЫ': 'ASTANA',  # genitive case
    'АЛМАТЫ': 'ALMATY',
    'ШЫМКЕНТ': 'SHYMKENT',
    'КАРАГАНДА': 'KARAGANDA',
    'ТАРАЗ': 'TARAZ',
    'ПАВЛОДАР': 'PAVLODAR',
    'СЕМЕЙ': 'SEMEY',
    'КОСТАНАЙ': 'KOSTANAY',
    'УСТЬ-КАМЕНОГОРСК': 'UST-KAMENOGORSK',
    'ПЕТРОПАВЛОВСК': 'PETROPAVLOVSK',
    'АТЫРАУ': 'ATYRAU',
    'ОРАЛ': 'ORAL',
    'УРАЛЬСК': 'URALSK',
    'КЫЗЫЛОРДА': 'KYZYLORDA',
    'ТАЛДЫКОРГАН': 'TALDYKORGAN',
    'АРКАЛЫК': 'ARKALYK',
    'АКТОБЕ': 'AKTOBE',
    'ЭКИБАСТУЗ': 'EKIBASTUZ',
    'РУДНЫЙ': 'RUDNY',
    'ЖЕЗКАЗГАН': 'ZHEZKAZGAN',
    'ТУРКЕСТАН': 'TURKESTAN',
    'НУР-СУЛТАН': 'ASTANA',
}

# NUR-SULTAN → ASTANA
_CITY_NORMALIZE = {
    'NUR-SULTAN': 'ASTANA',
}

# All known city tokens (for regex building)
_ALL_CITY_TOKENS = sorted(
    list(_CITIES_EN) + list(_CITIES_RU_MAP.keys()),
    key=len, reverse=True,
)

# Trailing city regex: city as last word(s), optionally preceded by "г."
_TRAILING_CITY_RE = re.compile(
    r'(?:\s+г\.)?'
    r'\s+(' + '|'.join(re.escape(c) for c in _ALL_CITY_TOKENS) + r')\s*$',
    flags=re.IGNORECASE,
)

# Known place names that should move from business name into address
# Longer patterns first so they match before shorter substrings.
KNOWN_PLACES = [
    'ТРЦ MEGA SILKWAY', 'ТРЦ MEGA SILK WAY',
    'ТРЦ МЕГА СИЛКВАЙ', 'ТЦ МЕГА СИЛКВАЙ',
    'MEGA SILKWAY', 'MEGA SILK WAY', 'МЕГА СИЛКВАЙ',
    'НАЗАРБАЕВ УНИВЕРСИТЕТ',
    'ТРЦ ЕВРАЗИЯ', 'ТРЦ KHAN SHATYR', 'ТРЦ ХАН ШАТЫР',
    'ТРЦ DOSTYK PLAZA', 'TC DOSTYK PLAZA', 'ТРЦ ESENTAI MALL',
    'ТРЦ KERUEN', 'ТРЦ КЕРУЕН',
    'DOSTYK PLAZA', 'ESENTAI MALL',
    'KHAN SHATYR', 'ХАН ШАТЫР', 'AZIYA PARK',
    'KERUEN', 'КЕРУЕН', 'МЕДЕО', 'MEDEO',
    'ТРЦ MEGA', 'ТРЦ МЕГА', 'ТЦ МЕГА', 'ТС MEGA', 'ТС МЕГА',
    'MEGA', 'МЕГА',
]

_TYPE_MAP = {'ИП': 'IP', 'ТОО': 'TOO'}

# Single-word places that are too generic to match at position 0
# (e.g. "Mega Coffee" should NOT extract "Mega" as a place).
_GENERIC_PLACES = {'MEGA', 'МЕГА'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_city(name: str) -> str:
    """Normalize a matched city token to canonical English upper-case."""
    upper = name.upper()
    # Russian → English
    if upper in _CITIES_RU_MAP:
        return _CITIES_RU_MAP[upper]
    # NUR-SULTAN → ASTANA
    if upper in _CITY_NORMALIZE:
        return _CITY_NORMALIZE[upper]
    return upper


# ---------------------------------------------------------------------------
# Main parsing
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
        city = _normalize_city(m_city.group(1))
        text = text[:m_city.start()].strip()

    # 2. Extract business type
    business_type = None
    m = BUSINESS_TYPE_RE.search(text)
    if m:
        business_type = m.group(1).upper()
        business_type = _TYPE_MAP.get(business_type, business_type)
        text = (text[:m.start()] + ' ' + text[m.end():]).strip()

    # 3. Clean up quotes/brackets/punctuation → replace with spaces
    text = re.sub(r'["\'\[\]\(\)«»,;.]', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Known places → split into address
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
                                and place in _GENERIC_PLACES):
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

    # If extracting the known place left business_name empty,
    # keep the place as both business_name and address
    if business_name is None and address is not None:
        business_name = address

    return business_type, business_name, address, city


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
    df['address'] = [x[2] for x in parsed]
    df['city'] = [x[3] for x in parsed]

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    purch = df[df['operation'].isin(_PURCHASE_OPS)]
    print(f"Done: {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Purchases/Withdrawal: {len(purch)}")
    print(f"  business_type filled: {purch['business_type'].notna().sum()}")
    print(f"  business_name filled: {purch['business_name'].notna().sum()}")
    print(f"  address filled: {purch['address'].notna().sum()}")
    print(f"  city filled: {purch['city'].notna().sum()}")

    return df


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

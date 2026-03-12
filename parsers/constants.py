"""Shared constants for bank statement normalizers."""

import re

# ---------------------------------------------------------------------------
# Business type detection
# ---------------------------------------------------------------------------

BUSINESS_TYPE_RE = re.compile(
    r'\b(ИП|ТОО|TOO|IP|IPP|LLP|LLC)\b\.?', flags=re.IGNORECASE
)

# Russian business type → canonical English abbreviation
TYPE_MAP = {'ИП': 'IP', 'ТОО': 'TOO'}

# ---------------------------------------------------------------------------
# Kazakhstan cities — English canonical forms
# ---------------------------------------------------------------------------

CITIES_EN = {
    'ASTANA', 'ALMATY', 'SHYMKENT', 'KARAGANDA', 'TARAZ', 'PAVLODAR',
    'SEMEY', 'KOSTANAY', 'UST-KAMENOGORSK', 'PETROPAVLOVSK', 'ATYRAU',
    'ORAL', 'URALSK', 'KYZYLORDA', 'TALDYKORGAN', 'ARKALYK', 'AKTOBE',
    'EKIBASTUZ', 'RUDNY', 'ZHEZKAZGAN', 'TURKESTAN',
    'NUR-SULTAN',
    'ARYS', 'AKKOL',
}

# Russian city names → canonical English upper-case
CITIES_RU_MAP = {
    'АСТАНА': 'ASTANA',
    'АСТАНЫ': 'ASTANA',
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
CITY_NORMALIZE = {
    'NUR-SULTAN': 'ASTANA',
}

# ---------------------------------------------------------------------------
# Known places (malls, universities, etc.)
# Longer patterns first so they match before shorter substrings.
# ---------------------------------------------------------------------------

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
    'DOSTYK',
]

# Single-word places too generic to match at word position 0
# (e.g. "Mega Coffee" should NOT extract "Mega" as a place).
GENERIC_PLACES = {'MEGA', 'МЕГА'}

# ---------------------------------------------------------------------------
# Common regex patterns for text cleanup
# ---------------------------------------------------------------------------

# Foreign currency amounts e.g. "(- 36,49 CNY)"
FOREIGN_AMOUNT_RE = re.compile(r'\([-–]\s*[\d\s,.]+[A-Z]{3}\)')

# Punctuation to replace with spaces
PUNCTUATION_RE = re.compile(r'["\'\[\]\(\)«»,;.*_#!/]')

# Hyphen before digit → space  (e.g. KASSA-1 → KASSA 1)
HYPHEN_DIGIT_RE = re.compile(r'-(\d)')

# Trailing " KZ" domain suffix
TRAILING_KZ_RE = re.compile(r'\s+KZ$', flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def normalize_city(name: str) -> str:
    """Normalize a city name to canonical English upper-case."""
    upper = name.upper()
    if upper in CITIES_RU_MAP:
        return CITIES_RU_MAP[upper]
    if upper in CITY_NORMALIZE:
        return CITY_NORMALIZE[upper]
    return upper


def strip_foreign_amount(text: str) -> str:
    """Remove foreign currency amounts like '(- 36,49 CNY)' from text."""
    return FOREIGN_AMOUNT_RE.sub('', text).strip()


def strip_punctuation(text: str) -> str:
    """Replace quotes, brackets, and common punctuation with spaces."""
    return PUNCTUATION_RE.sub(' ', text)


def normalize_hyphen_digit(text: str) -> str:
    """Convert hyphen-digit to space-digit (e.g. KASSA-1 → KASSA 1)."""
    return HYPHEN_DIGIT_RE.sub(r' \1', text)


def strip_digit_tokens(name: str) -> str:
    """Remove purely-digit tokens from a business name."""
    if not name:
        return name
    words = [w for w in name.split() if not w.isdigit()]
    return ' '.join(words).strip() or None


def strip_trailing_kz(name: str) -> str:
    """Strip trailing ' KZ' domain suffix (e.g. OZON.KZ → OZON KZ → OZON)."""
    if name and name.upper().endswith(' KZ'):
        return name[:-3].strip() or None
    return name

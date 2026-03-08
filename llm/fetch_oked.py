"""Fetch OKED codes from kompra.kz API for normalized bank statement CSVs."""

import argparse
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# City / business-type translation (English → Russian for kompra.kz)
# ---------------------------------------------------------------------------

TRANSLATED_CITIES = {
    'ASTANA': 'Астана',
    'ALMATY': 'Алматы',
    'SHYMKENT': 'Шымкент',
    'KARAGANDA': 'Караганда',
    'TARAZ': 'Тараз',
    'PAVLODAR': 'Павлодар',
    'SEMEY': 'Семей',
    'KOSTANAY': 'Костанай',
    'UST-KAMENOGORSK': 'Усть-Каменогорск',
    'PETROPAVLOVSK': 'Петропавловск',
    'ATYRAU': 'Атырау',
    'ORAL': 'Орал',
    'URALSK': 'Уральск',
    'KYZYLORDA': 'Кызылорда',
    'TALDYKORGAN': 'Талдыкорган',
    'AKTOBE': 'Актобе',
    'EKIBASTUZ': 'Екибастуз',
    'RUDNY': 'Рудный',
    'ZHEZKAZGAN': 'Жезказган',
    'TURKESTAN': 'Туркестан',
    'ARKALYK': 'Аркалык',
}

TRANSLATED_TYPES = {
    'IP': 'ИП',
    'TOO': 'ТОО',
    'LLC': 'ООО',
    'LLP': 'ООО',
}

# Operations that correspond to business purchases
_PURCHASE_OPS = {'Purchases', 'Withdrawal', 'Acquiring', 'Pending amount'}

# Known online / digital services — skip kompra.kz lookup, assign label directly.
# Matched by prefix (case-insensitive) against business_name.
ONLINE_SERVICES = [
    ('SPOTIFY', 'Online streaming service'),
    ('COURSERA', 'Online education platform'),
    ('CANVA', 'Online design platform'),
    ('NETFLIX', 'Online video streaming service'),
    ('GOOGLE', 'Online technology services'),
    ('APPLE', 'Online technology services'),
    ('STEAMGAMES', 'Online gaming platform'),
    # YANDEX — differentiate by sub-service (longer prefixes first)
    ('YANDEX EDA', 'Рестораны и услуги по доставке продуктов питания'),
    ('YANDEX DELIVERY', 'Услуги доставки'),
    ('YANDEX GO', 'Услуги такси и перевозок'),
    ('YANDEX PLUS', 'Онлайн подписка'),
    ('YANDEX', 'Online technology services'),
    ('WOLT', 'Рестораны и услуги по доставке продуктов питания'),
    ('CURSOR', 'Software development tool'),
]


def _match_online_service(business_name: str) -> str | None:
    """Return service description if business_name matches a known online service."""
    if not business_name:
        return None
    upper = business_name.upper().strip()
    for prefix, desc in ONLINE_SERVICES:
        if upper.startswith(prefix):
            return desc
    return None


# ---------------------------------------------------------------------------
# BLEU-like similarity scoring
# ---------------------------------------------------------------------------

def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Character-level n-gram (1-4) similarity score between 0 and 1."""
    if not reference or not candidate:
        return 0.0

    ref = reference.upper().strip()
    cand = candidate.upper().strip()

    if ref == cand:
        return 1.0

    precisions = []
    for n in range(1, 5):
        ref_ngrams = [ref[i:i + n] for i in range(len(ref) - n + 1)]
        cand_ngrams = [cand[i:i + n] for i in range(len(cand) - n + 1)]

        if not cand_ngrams:
            precisions.append(0.0)
            continue

        ref_counts: dict[str, int] = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1

        matches = 0
        for ng in cand_ngrams:
            if ng in ref_counts and ref_counts[ng] > 0:
                matches += 1
                ref_counts[ng] -= 1

        precisions.append(matches / len(cand_ngrams))

    if all(p > 0 for p in precisions):
        geo_mean = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    else:
        non_zero = [p for p in precisions if p > 0]
        geo_mean = sum(non_zero) / len(non_zero) if non_zero else 0.0

    # Brevity penalty
    bp = 1.0 if len(cand) >= len(ref) else min(1.0, (len(cand) / len(ref)) ** 0.5)

    return bp * geo_mean


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def calculate_match_score(business: tuple, searched: dict) -> float:
    """Score a kompra.kz result against our business tuple (name, type, city)."""
    bname, btype, bcity = business

    # Translate to Russian for comparison
    bcity_ru = TRANSLATED_CITIES.get(str(bcity).upper(), bcity) if pd.notna(bcity) else None
    btype_ru = TRANSLATED_TYPES.get(str(btype).upper(), btype) if pd.notna(btype) else None

    s_name = searched.get('name', '')
    s_location = searched.get('location', '')

    # Extract business type from searched name
    m = re.search(r'\b(ИП|ЮЛ|ТОО|ООО|ОАО|ЗАО|НПО)\b\.?', s_name)
    s_type = m.group(1).upper() if m else None

    score = 0.0

    # Name similarity — weight 0.7
    if bname and s_name:
        score += calculate_bleu_score(bname.upper(), s_name.upper()) * 0.7

    # City match — weight 0.2
    if bcity_ru and s_location:
        if bcity_ru.upper() in s_location.upper() or s_location.upper() in bcity_ru.upper():
            score += 0.2

    # Business type match — weight 0.1
    if btype_ru and s_type and btype_ru.upper() == s_type:
        score += 0.1

    return score


# ---------------------------------------------------------------------------
# kompra.kz API
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Content-Type': 'application/json',
        'Origin': 'https://kompra.kz',
        'Referer': 'https://kompra.kz/',
    })
    return session


def search_kompra_api(session: requests.Session, query: str, search_type: str = 'ul') -> list[dict]:
    """Search kompra.kz API. search_type: 'ip' or 'ul'."""
    url = f"https://gateway.kompra.kz/search?type={search_type}"
    payload = {"page": 1, "limit": 50, "name": query}

    try:
        resp = session.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get('content', [])[:10]:
            try:
                r = {
                    'name': item.get('name', ''),
                    'bin': item.get('identifier', ''),
                    'location': item.get('law_address', ''),
                    'oked_code': item.get('oked', {}).get('code', ''),
                    'oked_description': item.get('oked', {}).get('name_ru', ''),
                }
                if r['name'] and r['bin']:
                    results.append(r)
            except Exception:
                continue
        return results
    except Exception as e:
        print(f"    API error ({search_type}): {str(e)[:80]}")
        return []


def search_and_extract_oked(session: requests.Session, business: tuple) -> dict | None:
    """Search kompra.kz for a business and return best OKED match."""
    bname, btype, bcity = business
    print(f"  Business: name={bname}, type={btype}, city={bcity}")

    all_results = []

    print(f"    Searching ЮЛ...")
    all_results.extend(search_kompra_api(session, bname, 'ul'))
    time.sleep(1)

    print(f"    Searching ИП...")
    all_results.extend(search_kompra_api(session, bname, 'ip'))

    if not all_results:
        print(f"    No results found")
        return None

    print(f"    Found {len(all_results)} total results")

    best_match = None
    best_score = 0.0

    for result in all_results:
        score = calculate_match_score(business, result)
        if score > best_score:
            best_score = score
            best_match = result

    if not best_match or best_score < 0.3:
        print(f"    No good match (best score: {best_score:.2f})")
        return None

    print(f"    Best match: {best_match['name']} (score: {best_score:.2f})")

    oked_desc = best_match.get('oked_description', '')
    oked_code = best_match.get('oked_code', '')

    if oked_desc:
        print(f"    OKED: {oked_code or 'N/A'} — {oked_desc[:60]}...")
        return {'code': oked_code or None, 'description': oked_desc}

    print(f"    No OKED data found")
    return None


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def _fetch_one(idx: int, total: int, key: tuple, print_lock: threading.Lock) -> tuple[tuple, dict | None]:
    """Worker: fetch OKED for a single business. Each thread gets its own session."""
    bname, btype, bcity = key

    # Check for known online/digital services first
    online_desc = _match_online_service(bname)
    if online_desc:
        with print_lock:
            print(f"[{idx}/{total}] Online service: {bname} → {online_desc}")
        return key, {'code': 'ONLINE', 'description': online_desc}

    session = _make_session()
    with print_lock:
        print(f"[{idx}/{total}] Processing: name={bname}, type={btype}, city={bcity}")
    oked = search_and_extract_oked(session, key)
    return key, oked


def process_dataframe(df: pd.DataFrame, workers: int = 5) -> pd.DataFrame:
    """Fetch OKED codes for a DataFrame (in-process, no file I/O).

    Adds ``oked_code`` and ``oked_description`` columns and returns the
    modified DataFrame.
    """
    print(f"  Total rows: {len(df)}")

    # Filter for purchase operations
    purchase_mask = df['operation'].isin(_PURCHASE_OPS)
    purchase_df = df[purchase_mask]
    print(f"  Purchase rows: {len(purchase_df)}")

    # Get unique (business_name, business_type, city) tuples
    biz_cols = ['business_name', 'business_type', 'city']
    biz_df = purchase_df[biz_cols].dropna(subset=['business_name']).drop_duplicates()
    total = len(biz_df)
    print(f"  Unique businesses to search: {total}")
    print(f"  Workers: {workers}\n")

    keys = [
        (row['business_name'], row['business_type'], row['city'])
        for _, row in biz_df.iterrows()
    ]

    oked_code_map: dict[tuple, str | None] = {}
    oked_desc_map: dict[tuple, str | None] = {}
    found = 0
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, i, total, key, print_lock): key
            for i, key in enumerate(keys, 1)
        }

        for future in as_completed(futures):
            key, oked = future.result()
            if oked:
                oked_code_map[key] = oked.get('code')
                oked_desc_map[key] = oked.get('description')
                found += 1
            else:
                oked_code_map[key] = None
                oked_desc_map[key] = None

    print(f"\n{'=' * 70}")
    print(f"Summary:")
    print(f"  Unique businesses searched: {total}")
    print(f"  OKED codes found: {found}")
    print(f"  Not found: {total - found}")
    print(f"{'=' * 70}\n")

    # Map back to all rows
    def _get_code(row):
        if row['operation'] not in _PURCHASE_OPS:
            return None
        key = (row.get('business_name'), row.get('business_type'), row.get('city'))
        return oked_code_map.get(key)

    def _get_desc(row):
        if row['operation'] not in _PURCHASE_OPS:
            return None
        key = (row.get('business_name'), row.get('business_type'), row.get('city'))
        return oked_desc_map.get(key)

    df['oked_code'] = df.apply(_get_code, axis=1)
    df['oked_description'] = df.apply(_get_desc, axis=1)

    return df


def process_csv(input_file: str, output_file: str, workers: int = 5) -> pd.DataFrame:
    """Read normalized CSV, fetch OKED for purchase rows (multithreaded), save result."""
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Total rows: {len(df)}")

    # Filter for purchase operations
    purchase_mask = df['operation'].isin(_PURCHASE_OPS)
    purchase_df = df[purchase_mask]
    print(f"  Purchase rows: {len(purchase_df)}")

    # Get unique (business_name, business_type, city) tuples — skip rows without business_name
    biz_cols = ['business_name', 'business_type', 'city']
    biz_df = purchase_df[biz_cols].dropna(subset=['business_name']).drop_duplicates()
    total = len(biz_df)
    print(f"  Unique businesses to search: {total}")
    print(f"  Workers: {workers}\n")

    keys = [
        (row['business_name'], row['business_type'], row['city'])
        for _, row in biz_df.iterrows()
    ]

    oked_code_map: dict[tuple, str | None] = {}
    oked_desc_map: dict[tuple, str | None] = {}
    found = 0
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, i, total, key, print_lock): key
            for i, key in enumerate(keys, 1)
        }

        for future in as_completed(futures):
            key, oked = future.result()
            if oked:
                oked_code_map[key] = oked.get('code')
                oked_desc_map[key] = oked.get('description')
                found += 1
            else:
                oked_code_map[key] = None
                oked_desc_map[key] = None

    print(f"\n{'=' * 70}")
    print(f"Summary:")
    print(f"  Unique businesses searched: {total}")
    print(f"  OKED codes found: {found}")
    print(f"  Not found: {total - found}")
    print(f"{'=' * 70}\n")

    # Map back to all rows
    def _get_code(row):
        if row['operation'] not in _PURCHASE_OPS:
            return None
        key = (row.get('business_name'), row.get('business_type'), row.get('city'))
        return oked_code_map.get(key)

    def _get_desc(row):
        if row['operation'] not in _PURCHASE_OPS:
            return None
        key = (row.get('business_name'), row.get('business_type'), row.get('city'))
        return oked_desc_map.get(key)

    df['oked_code'] = df.apply(_get_code, axis=1)
    df['oked_description'] = df.apply(_get_desc, axis=1)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_file}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fetch OKED codes from kompra.kz for normalized bank CSVs")
    ap.add_argument("--input", required=True, help="Input normalized CSV (with business_name column)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--workers", type=int, default=5, help="Number of parallel threads (default: 5)")
    args = ap.parse_args()

    process_csv(args.input, args.out, workers=args.workers)


if __name__ == '__main__':
    main()

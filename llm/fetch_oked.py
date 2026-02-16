import requests
import pandas as pd
import time
import re

def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU-like score for string similarity.
    Uses character-level n-grams (1-4 grams) for robust matching.
    Returns score between 0 and 1.
    """
    if not reference or not candidate:
        return 0.0
    
    # Normalize strings
    ref = reference.upper().strip()
    cand = candidate.upper().strip()
    
    if ref == cand:
        return 1.0
    
    # Calculate n-gram precisions (1-gram to 4-gram)
    precisions = []
    
    for n in range(1, 5):  # 1-gram, 2-gram, 3-gram, 4-gram
        ref_ngrams = [ref[i:i+n] for i in range(len(ref)-n+1)]
        cand_ngrams = [cand[i:i+n] for i in range(len(cand)-n+1)]
        
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        
        # Count matches
        matches = 0
        ref_ngram_counts = {}
        for ngram in ref_ngrams:
            ref_ngram_counts[ngram] = ref_ngram_counts.get(ngram, 0) + 1
        
        for ngram in cand_ngrams:
            if ngram in ref_ngram_counts and ref_ngram_counts[ngram] > 0:
                matches += 1
                ref_ngram_counts[ngram] -= 1
        
        precision = matches / len(cand_ngrams)
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if all(p > 0 for p in precisions):
        geometric_mean = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    else:
        # If any precision is 0, use arithmetic mean of non-zero precisions
        non_zero = [p for p in precisions if p > 0]
        geometric_mean = sum(non_zero) / len(non_zero) if non_zero else 0.0
    
    # Brevity penalty (penalize length differences)
    ref_len = len(ref)
    cand_len = len(cand)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = min(1.0, (cand_len / ref_len) ** 0.5) if ref_len > 0 else 0.0
    
    bleu = bp * geometric_mean
    
    return bleu

translated_cities = {
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
Translated_business_types = {
    'IP': 'ИП',
    'UL': 'ЮЛ',
    'TOO': 'ТОО',
    'OOO': 'ООО',
    'LLC': 'ООО',
    'LLP': 'ООО',
    'NPO': 'НПО',
}

def calculate_match_score(business, searched_business):
    """
    Calculate match score based on name and city similarity using BLEU score.
    Returns score between 0 and 1.
    """
    business_name, business_type, business_city = business[0], business[1], business[2]
    
    # translate city name to Russian, because kompra.kz uses Russian city names
    # Check for pandas NaN values using pd.notna()
    business_city = translated_cities[business_city] if pd.notna(business_city) else None
    
    # translate business type to Russian, because kompra.kz uses Russian business types
    business_type = Translated_business_types[business_type] if pd.notna(business_type) else None
    
    searched_business_name = searched_business['name']
    searched_business_city = searched_business['location']
    
    # Extract business type from searched business name (handle case where no match is found)
    type_match = re.search(r'\b(ИП|ЮЛ|ТОО|ООО|ОАО|ЗАО|НПО)\b\.?', searched_business_name)
    searched_business_type = type_match.group(1).upper() if type_match else None
    
    score = 0.0
    
    # Name similarity using BLEU score - weight 0.7
    if business_name and searched_business_name:
        # Clean names (remove common prefixes/suffixes)
        business_name_clean = business_name.upper().strip()
        searched_business_name_clean = searched_business_name.upper().strip()
        
        # Calculate BLEU score
        name_similarity = calculate_bleu_score(business_name_clean, searched_business_name_clean)
        
        score += name_similarity * 0.7
    
    # City match - weight 0.2
    if business_city is not None and searched_business_city is not None:
        business_city_clean = business_city.upper().strip()
        searched_business_city_clean = searched_business_city.upper().strip()
        
        if business_city_clean in searched_business_city_clean or searched_business_city_clean in business_city_clean:
            # print(f"  City match: {business_city_clean} == {searched_business_city_clean}")
            score += 0.2
    
    # Business type match - weight 0.1
    # usually komrpa.kz includes business type in the name
    if business_type is not None and searched_business_type is not None and business_type == searched_business_type:
        # print(f"  Business type match: {business_type} == {searched_business_type}")
        score += 0.1
    
    return score


def search_kompra_api(session, query, search_type='ul'):
    """
    Search kompra.kz using their API.
    search_type: 'ip' for ИП or 'ul' for ЮЛ
    Returns list of dicts with company info including OKED.
    """
    api_url = f"https://gateway.kompra.kz/search?type={search_type}"
    
    payload = {
        "page": 1,
        "limit": 50,
        "name": query
    }
    
    try:
        response = session.post(api_url, json=payload, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        # Parse the 'content' array from the API response
        for item in data.get('content', [])[:10]:  # Take first 10 results
            try:
                result = {
                    'name': item.get('name', ''),
                    'bin': item.get('identifier', ''),
                    'location': item.get('law_address', ''),
                    'oked_code': item.get('oked', {}).get('code', ''),
                    'oked_description': item.get('oked', {}).get('name_ru', '')
                }
                
                if result['name'] and result['bin']:
                    results.append(result)
                    
            except Exception as e:
                continue
        
        return results
        
    except Exception as e:
        print(f"    API error ({search_type}): {str(e)[:50]}")
        return []

def search_and_extract_oked(session, business):
    """
    Search kompra.kz API and extract OKED codes for given business (Business Name and City).
    Returns dict with OKED code and description or None.
    """
    print(f"  Business: {business}")
    business_name, business_type, city = business
        
    # Search in both IP and UL modes
    all_results = []
    
    # Search UL (legal entities)
    print(f"    Searching ЮЛ...")
    ul_results = search_kompra_api(session, business_name, 'ul')
    all_results.extend(ul_results)
    
    time.sleep(1)  # Small delay between searches
    
    # Search IP (individual entrepreneurs)
    print(f"    Searching ИП...")
    ip_results = search_kompra_api(session, business_name, 'ip')
    all_results.extend(ip_results)
    
    if not all_results:
        print(f"  No results found")
        return None
    
    print(f"  Found {len(all_results)} total results")
    
    # Score and find best match
    best_match = None
    best_score = 0.0
    
    for result in all_results:

        score = calculate_match_score(
            business,
            result
        )

        # print(f"  Result: {result['name']}")
        # print(f"  Score: {score}")
        # print(f"  Business Name: {business_name}")
        # print(f"  City: {city}")
        # print(f"  Location: {result['location']}")
        # print("\n")
        
        if score > best_score:
            best_score = score
            best_match = result
    
    if not best_match or best_score < 0.3:  # Minimum threshold
        print(f"  No good match (best score: {best_score:.2f})")
        return None
    
    print(f"  Best match: {best_match['name']} (score: {best_score:.2f})")
    print(f"  БИН: {best_match['bin']}")
    
    # Extract OKED from the matched result
    oked_code = best_match.get('oked_code', '')
    oked_description = best_match.get('oked_description', '')
    
    if oked_description:
        print(f"  OKED: {oked_code if oked_code else 'N/A'} - {oked_description[:60]}...")
        return {
            'code': oked_code if oked_code else None,
            'description': oked_description
        }
    else:
        print(f"  No OKED data found")
        return None


def process_bank_statements(input_file, output_file):
    """
    Read bank statements, fetch OKED codes from kompra.kz API for Acquiring operations,
    and save results with new OKED Code and OKED Description columns.
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df)}")
    
    # Filter for Acquiring operations
    acquiring_df = df[df['Operation'] == 'Acquiring'].copy()
    print(f"Acquiring operations: {len(acquiring_df)}")
    
    # Get unique Details values (non-empty)
    unique_businesses = acquiring_df[['Business Name', 'Business Type', 'City']].drop_duplicates()
    print(f"Unique Businesses: {unique_businesses.head()}")
    print(f"Unique Businesses to search: {len(unique_businesses)}\n")
    
    # Create session for persistent connection
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Content-Type': 'application/json',
        'Origin': 'https://kompra.kz',
        'Referer': 'https://kompra.kz/'
    })
    
    # Create mapping: Details -> OKED data
    oked_code_mapping = {}
    oked_desc_mapping = {}
    found_count = 0
    
    for count, (idx, row) in enumerate(unique_businesses.iterrows(), start=1):
        # Convert row (Series) to tuple
        business = tuple(row)
        print(f"[{count}/{len(unique_businesses)}] Processing: {business}")
        oked_data = search_and_extract_oked(session, business)
        
        if oked_data:
            oked_code_mapping[business] = oked_data.get('code')
            oked_desc_mapping[business] = oked_data.get('description')
            found_count += 1
        else:
            oked_code_mapping[business] = None
            oked_desc_mapping[business] = None
        
        # Be respectful with requests - add delay
        if count < len(unique_businesses):
            time.sleep(3)
    
    # Map OKED codes to all rows
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Unique Businesses searched: {len(unique_businesses)}")
    print(f"  OKED codes found: {found_count}")
    print(f"  Not found: {len(unique_businesses) - found_count}")
    print(f"{'='*70}\n")
    
    # Add OKED Code and OKED Description columns to original dataframe
    # Map using tuple of (Business Name, Business Type, City) to match the keys
    def get_oked_code(row):
        if row['Operation'] not in ['Acquiring', 'Pending amount']:
            return None
        key = (row['Business Name'], row['Business Type'], row['City'])
        return oked_code_mapping.get(key)
    
    def get_oked_desc(row):
        if row['Operation'] not in ['Acquiring', 'Pending amount']:
            return None
        key = (row['Business Name'], row['Business Type'], row['City'])
        return oked_desc_mapping.get(key)
    
    df['OKED Code'] = df.apply(get_oked_code, axis=1)
    df['OKED Description'] = df.apply(get_oked_desc, axis=1)
    
    # Save to new CSV
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved results to: {output_file}")
    
    return df


if __name__ == '__main__':
    # Test with single example first
    print("="*70)
    print("OKED Code Fetcher from kompra.kz - TEST MODE")
    print("="*70 + "\n")
    
    # Create session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Content-Type': 'application/json',
        'Origin': 'https://kompra.kz',
        'Referer': 'https://kompra.kz/'
    })
    
    # Test with one example
    # test_details = "PINGWIN PREMIUM BOULING ASTANA KZ"
    
    # print(f"Testing with: {test_details}\n")
    
    # result = search_and_extract_oked(session, test_details)
    
    # print(f"\n{'='*70}")
    # print(f"Result: {result}")
    # print(f"{'='*70}\n")
    
    # Uncomment below to process full file
    input_file = r'C:\Users\Lenovo\Desktop\hw-nlp\bank-classifier\bank_statements_normalized.csv'
    output_file = r'C:\Users\Lenovo\Desktop\hw-nlp\bank-classifier\bank_statements_with_oked.csv'
    process_bank_statements(input_file, output_file)


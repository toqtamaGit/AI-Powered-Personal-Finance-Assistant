import pandas as pd
import re
from difflib import SequenceMatcher

# List of major cities in Kazakhstan
KAZAKHSTAN_CITIES = [
    'ASTANA', 'ALMATY', 'SHYMKENT', 'KARAGANDA', 'TARAZ', 'PAVLODAR',
    'SEMEY', 'KOSTANAY', 'UST-KAMENOGORSK', 'PETROPAVLOVSK', 'ATYRAU',
    'ORAL', 'URALSK', 'KYZYLORDA', 'TALDYKORGAN', 'ARKALYK', 'AKTOBE',
    'EKIBASTUZ', 'RUDNY', 'ZHEZKAZGAN', 'TURKESTAN'
]

# List of known public places (malls, buildings)
KNOWN_PLACES = [
    'MEGA SILKWAY', 'MEGA SILK WAY', 'DOSTYK PLAZA', 'AZIYA PARK',
    'MEGA', 'DOSTYK',
]

# Cache for unique details to build reference list for fuzzy matching
_reference_details = set()

def normalize_city(text):
    """Normalize city names in the address text."""
    if not text:
        return text
    
    # Apply city name normalizations - all variations to Astana
    text = re.sub(r'\bNUR-SULTAN\b', 'ASTANA', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAST\s*[>]?\s*NA\b', 'ASTANA', text, flags=re.IGNORECASE)
    text = re.sub(r'\bNUR-SU\s*\w*\s*TAN\b', 'ASTANA', text, flags=re.IGNORECASE)
    text = re.sub(r'\bULTAN\b', 'ASTANA', text, flags=re.IGNORECASE)
    
    return text.strip()

def fuzzy_match_details(details):
    """
    Apply fuzzy matching to correct typos in details string.
    Looks for similar strings in reference list with 90% similarity threshold.
    """
    if not details or not _reference_details:
        return details
    
    best_match = None
    best_ratio = 0.9  # 90% threshold
    
    for reference in _reference_details:
        # Calculate similarity ratio
        ratio = SequenceMatcher(None, details.upper(), reference.upper()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = reference
    
    # Return best match if found, otherwise original
    return best_match if best_match else details

def extract_city_from_text(text):
    """
    Extract city name from text by matching against known cities.
    Returns the normalized city name or None.
    """
    if not text:
        return None
    
    text_upper = text.upper()
    
    # Check for each known city
    for city in KAZAKHSTAN_CITIES:
        if city in text_upper:
            return city.upper()
    
    if re.search(r'NUR-SULTAN|AST\s*[>]?\s*NA|NUR-SU\s*\w*\s*TAN|ULTAN', text_upper):
        return 'ASTANA'
    
    return None

def parse_details(details, operation):
    """
    Parse the Details column to extract business type, name, address, and city.
    Only processes rows where operation is 'Acquiring' or 'Pending amount'.
    
    Returns: (business_type, business_name, address, city)
    """
    # Initialize empty values
    business_type = None
    business_name = None
    address = None
    city = None
    
    # Only process Acquiring or Pending amount operations
    if operation not in ['Acquiring', 'Pending amount']:
        return business_type, business_name, address, city
    
    if not details or pd.isna(details):
        return business_type, business_name, address, city
    
    # Step 1: Apply fuzzy matching first to correct typos
    details = fuzzy_match_details(details)
    
    # Store in reference set for future fuzzy matching
    _reference_details.add(details)
    
    # Step 2: Extract business type (can appear anywhere, not just at start)
    business_type_pattern = r'\b(IP|TOO|ТОО|IPP|LLP|LLC)\b\.?'
    type_match = re.search(business_type_pattern, details, flags=re.IGNORECASE)
    
    if type_match:
        business_type = type_match.group(1).upper()
        # Remove the business type from details for further processing
        details_without_type = details[:type_match.start()] + details[type_match.end():]
        details_without_type = details_without_type.strip()
    else:
        details_without_type = details
    
    # Step 3: Clean text - remove quotes and brackets
    # Handle nested quotes like TOO ""KIKIS MS""
    cleaned = re.sub(r'["\'\[\]\(\)]', '', details_without_type)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Step 4: Determine address boundary
    words = cleaned.split()
    if len(words) == 0:
        return business_type, business_name, address, city
    
    # Check if "KZ" is present
    has_kz = 'KZ' in [w.upper() for w in words]
    
    if has_kz:
        # Address = last 2 words
        if len(words) >= 2:
            address_words = words[-2:]
            name_words = words[:-2]
        else:
            address_words = words
            name_words = []
    else:
        # Address = last 1 word (city only)
        if len(words) >= 1:
            address_words = words[-1:]
            name_words = words[:-1]
        else:
            address_words = []
            name_words = words
    
    # Check if name ends with a known place that should be included in address
    # Only check the end of name_words to see if it matches a known place
    if name_words:
        name_text = ' '.join(name_words).upper()
        found_place = False
        
        for place in KNOWN_PLACES:
            # Check if the known place appears at the end of the name
            if name_text.endswith(place):
                # Find how many words make up this place
                place_words = place.split()
                place_word_count = len(place_words)
                
                # Move only those specific place words from name to address
                if len(name_words) >= place_word_count:
                    place_from_name = name_words[-place_word_count:]
                    address_words = place_from_name + address_words
                    name_words = name_words[:-place_word_count]
                    found_place = True
                    break
            # Also check if place appears anywhere in the last few words before address
            elif place in name_text:
                # Find the position where this place starts
                for i in range(len(name_words)):
                    potential_phrase = ' '.join(name_words[i:]).upper()
                    if potential_phrase.startswith(place):
                        # Move words from this position to address
                        address_words = name_words[i:] + address_words
                        name_words = name_words[:i]
                        found_place = True
                        break
                if found_place:
                    break
    
    # Step 5: Extract business name
    business_name = ' '.join(name_words).strip() if name_words else None
    
    # Step 6: Normalize city and extract from address
    address = ' '.join(address_words).strip() if address_words else None
    
    if address:
        # Extract city from address
        city = extract_city_from_text(address)
    
    return business_type, business_name, address, city

def normalize_csv(input_file, output_file):
    """
    Read the bank CSV, normalize the Details column, and save to a new file.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Apply parsing to each row
    parsed_data = df.apply(
        lambda row: parse_details(row['Details'], row['Operation']), 
        axis=1
    )
    
    # Split the results into separate columns
    df['Business Type'] = [x[0] for x in parsed_data]
    df['Business Name'] = [x[1] for x in parsed_data]
    df['Address'] = [x[2] for x in parsed_data]
    df['City'] = [x[3] for x in parsed_data]
    
    # Save to new CSV
    df.to_csv(output_file, index=False)
    print(f"Normalized CSV saved to: {output_file}")
    print(f"Total rows processed: {len(df)}")
    print(f"Rows with parsed data: {len(df[df['Business Type'].notna()])}")

if __name__ == '__main__':
    input_file = r'C:\Users\Lenovo\Desktop\hw-nlp\bank-classifier\bank_statements.csv'
    output_file = r'C:\Users\Lenovo\Desktop\hw-nlp\bank-classifier\bank_statements_normalized.csv'
    
    normalize_csv(input_file, output_file)


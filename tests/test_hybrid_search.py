"""
Test hybrid search to verify text matching prioritization
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load everything
print("Loading phrasebook...")
df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')
# Rename Francais column to French for consistency
if 'Francais' in df.columns:
    df.rename(columns={'Francais': 'French'}, inplace=True)
df['French'] = df['French'].astype(str).str.strip()

print("Loading models...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
faiss_index = faiss.read_index('assets/faiss_index.bin')

def find_closest_match(text):
    """Hybrid search implementation"""
    if df.empty:
        return None
    
    query_lower = text.lower().strip()
    
    # Step 1: Exact match
    exact_matches = df[df['French'].str.lower().str.strip() == query_lower]
    if not exact_matches.empty:
        result = exact_matches.iloc[0].to_dict()
        result['match_score'] = 100
        result['match_type'] = 'exact'
        return result
    
    # Step 2: Contains match
    contains_matches = df[df['French'].str.lower().str.contains(query_lower, na=False, regex=False)]
    if not contains_matches.empty:
        contains_matches = contains_matches.copy()
        contains_matches['length'] = contains_matches['French'].str.len()
        contains_matches = contains_matches.sort_values('length')
        result = contains_matches.iloc[0].to_dict()
        result['match_score'] = 95
        result['match_type'] = 'contains'
        return result
    
    # Step 3: Token match
    query_tokens = set(query_lower.split())
    if query_tokens:
        def contains_all_tokens(french_text):
            french_tokens = set(french_text.lower().split())
            return query_tokens.issubset(french_tokens)
        
        token_matches = df[df['French'].apply(contains_all_tokens)]
        if not token_matches.empty:
            token_matches = token_matches.copy()
            token_matches['length'] = token_matches['French'].str.len()
            token_matches = token_matches.sort_values('length')
            result = token_matches.iloc[0].to_dict()
            result['match_score'] = 85
            result['match_type'] = 'token'
            return result
    
    # Step 4: Semantic search
    query_embedding = embedding_model.encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    k = 5
    distances, indices = faiss_index.search(query_embedding, k)
    
    best_idx = indices[0][0]
    best_score = float(distances[0][0])
    score_percentage = best_score * 100
    
    if best_score > 0.65:
        result = df.iloc[best_idx].to_dict()
        result['match_score'] = score_percentage
        result['match_type'] = 'semantic'
        return result
    
    return None

# Test cases
test_cases = [
    ("bonjour", "Should match 'bonjour' exactly, NOT 'bonsoir'"),
    ("Bonjour", "Should match 'bonjour' (case insensitive)"),
    ("merci", "Should match sentences containing 'merci'"),
    ("je m'appelle", "Should match exact phrase"),
    ("Je m'appelle Gainvier", "Should contain 'je m'appelle'"),
    ("merci beaucoup", "Should find 'merci beaucoup' or similar"),
]

print("\n" + "="*70)
print("HYBRID SEARCH TESTS")
print("="*70)

for query, expected in test_cases:
    print(f"\nQuery: '{query}'")
    print(f"Expected: {expected}")
    
    result = find_closest_match(query)
    
    if result:
        print(f"✓ Match type: {result['match_type'].upper()}")
        print(f"  French: '{result['French']}'")
        print(f"  Nufi: '{result['Nufi']}'")
        print(f"  Score: {result['match_score']:.1f}")
    else:
        print("✗ No match found")

# Test specific cases from requirements
print("\n" + "="*70)
print("SPECIFIC REQUIREMENT TESTS")
print("="*70)

print("\n1. 'merci' should match all sentences containing 'merci':")
merci_matches = df[df['French'].str.lower().str.contains('merci', na=False)]
print(f"   Found {len(merci_matches)} entries with 'merci':")
for idx, row in merci_matches.head(5).iterrows():
    print(f"   - {row['French']}")

print("\n2. 'Bonjour' should prioritize 'bonjour', not 'bonsoir':")
bonjour_result = find_closest_match("Bonjour")
print(f"   Match: '{bonjour_result['French']}' (type: {bonjour_result['match_type']})")
print(f"   ✓ Correct!" if 'bonjour' in bonjour_result['French'].lower() and 'bonsoir' not in bonjour_result['French'].lower() else "   ✗ Wrong!")

print("\n" + "="*70)
print("All tests completed!")
print("="*70)

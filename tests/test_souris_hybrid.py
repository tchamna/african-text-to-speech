"""
Test hybrid search with complex sentence: "la souris a faim"
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load phrasebook
print("Loading phrasebook...")
df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')
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
    
    # Step 2: Partial word match
    query_words = [w for w in query_lower.replace("'", " ").split() if len(w) >= 3]
    if query_words:
        for word in query_words[:3]:
            word_matches = df[df['French'].str.lower().str.contains(word, na=False, regex=False)]
            if not word_matches.empty:
                word_matches = word_matches.copy()
                word_matches['length'] = word_matches['French'].str.len()
                word_matches = word_matches.sort_values('length')
                result = word_matches.iloc[0].to_dict()
                result['match_score'] = 90
                result['match_type'] = 'partial'
                result['matched_word'] = word
                return result
    
    # Step 3: Semantic search
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

# Test with "la souris a faim"
print("\n" + "="*70)
print("Testing: 'la souris a faim'")
print("="*70)

result = find_closest_match("la souris a faim")

if result:
    print(f"\n✓ Match found!")
    print(f"  Type: {result['match_type'].upper()}")
    if 'matched_word' in result:
        print(f"  Matched on word: '{result['matched_word']}'")
    print(f"  French: {result['French']}")
    print(f"  Nufi: {result['Nufi']}")
    print(f"  Score: {result['match_score']:.1f}")
else:
    print("\n✗ No match found")

# Show what would match for each word
print("\n" + "="*70)
print("Individual word analysis:")
print("="*70)

words = ["souris", "faim"]
for word in words:
    matches = df[df['French'].str.lower().str.contains(word, na=False)]
    print(f"\n'{word}' appears in {len(matches)} phrases:")
    for _, row in matches.head(3).iterrows():
        print(f"  - {row['French']}")

# Test with "je vais bien"
print("\n" + "="*70)
print("Testing: 'je vais bien'")
print("="*70)

result = find_closest_match("je vais bien")

if result:
    print(f"\n✓ Match found!")
    print(f"  Type: {result['match_type'].upper()}")
    if 'matched_word' in result:
        print(f"  Matched on word: '{result['matched_word']}'")
    print(f"  French: {result['French']}")
    print(f"  Nufi: {result['Nufi']}")
    print(f"  Score: {result['match_score']:.1f}")
else:
    print("\n✗ No match found")

# Show what would match for each word in "je vais bien"
print("\n" + "="*70)
print("Individual word analysis for 'je vais bien':")
print("="*70)

words_bien = ["je", "vais", "bien"]
for word in words_bien:
    matches = df[df['French'].str.lower().str.contains(word, na=False)]
    print(f"\n'{word}' appears in {len(matches)} phrases:")
    for _, row in matches.head(3).iterrows():
        print(f"  - {row['French']}")

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Load the phrasebook
df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')
df.columns = ['French', 'Nufi']
df = df[df['French'].notna() & df['Nufi'].notna()]
df = df[df['French'].str.strip().ne('') & df['Nufi'].str.strip().ne('')]

# Load FAISS and models
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
faiss_index = faiss.read_index('assets/faiss_index.bin')
with open('assets/index_mapping.pkl', 'rb') as f:
    index_mapping = pickle.load(f)

def find_closest_match(text):
    text_lower = text.lower().strip()
    
    # 1. Exact match
    exact_matches = df[df['French'].str.lower().str.strip() == text_lower]
    if not exact_matches.empty:
        return exact_matches.iloc[0]['French'], exact_matches.iloc[0]['Nufi'], 100, "EXACT"
    
    # 2. Multi-word phrase match - prioritize matches with multiple query words
    query_words = [w for w in text_lower.replace("'", " ").split() if len(w) >= 3]
    if len(query_words) >= 2:
        # Try to find phrases containing multiple words from query
        for i in range(len(query_words), 1, -1):  # Start with all words, then decrease
            for j in range(len(query_words) - i + 1):
                word_combo = query_words[j:j+i]
                multi_matches = df.copy()
                for word in word_combo:
                    multi_matches = multi_matches[multi_matches['French'].str.lower().str.contains(word, na=False, regex=False)]
                
                if not multi_matches.empty:
                    # Calculate match quality: prioritize phrases starting with query words
                    multi_matches = multi_matches.copy()
                    multi_matches['starts_with'] = multi_matches['French'].str.lower().str.startswith(tuple(word_combo))
                    multi_matches['word_count'] = multi_matches['French'].str.split().str.len()
                    # Sort: first by whether it starts with query words, then by word count (prefer longer, more complete phrases)
                    multi_matches = multi_matches.sort_values(['starts_with', 'word_count'], ascending=[False, False])
                    score = 95 - (5 * (len(query_words) - i))
                    return multi_matches.iloc[0]['French'], multi_matches.iloc[0]['Nufi'], score, f"MULTI-WORD {word_combo}"
    
    # 3. Single-word partial match
    if query_words:
        for word in query_words:
            partial_matches = df[df['French'].str.lower().str.contains(word, regex=False, na=False)]
            if not partial_matches.empty:
                partial_matches = partial_matches.copy()
                partial_matches['starts_with'] = partial_matches['French'].str.lower().str.startswith(word)
                partial_matches['length'] = partial_matches['French'].str.len()
                # Prioritize phrases starting with the word, then by length
                partial_matches = partial_matches.sort_values(['starts_with', 'length'], ascending=[False, True])
                return partial_matches.iloc[0]['French'], partial_matches.iloc[0]['Nufi'], 85, f"SINGLE-WORD '{word}'"
    
    # 3. Contains match
    contains_matches = df[df['French'].str.lower().str.contains(text_lower, regex=False, na=False)]
    if not contains_matches.empty:
        return contains_matches.iloc[0]['French'], contains_matches.iloc[0]['Nufi'], 95, "CONTAINS"
    
    # 4. Token match (any word from input in any phrase)
    if words:
        for word in words:
            token_matches = df[df['French'].str.lower().str.split().apply(lambda x: word in x)]
            if not token_matches.empty:
                return token_matches.iloc[0]['French'], token_matches.iloc[0]['Nufi'], 85, f"TOKEN '{word}'"
    
    # 5. Semantic search
    query_embedding = embedding_model.encode([text], normalize_embeddings=True)
    distances, indices = faiss_index.search(query_embedding, k=1)
    
    if len(indices[0]) > 0:
        best_idx = indices[0][0]
        similarity_score = float(distances[0][0] * 100)
        
        if best_idx in index_mapping:
            matched_french = index_mapping[best_idx]
            matched_row = df[df['French'] == matched_french]
            if not matched_row.empty:
                return matched_french, matched_row.iloc[0]['Nufi'], similarity_score, "SEMANTIC"
    
    return None, None, 0, "NO MATCH"

# Test with "Je m'appelle Zona"
print("Testing: 'Je m'appelle Zona'")
french, nufi, score, match_type = find_closest_match("Je m'appelle Zona")
print(f"Result: {match_type} match - '{french}' → '{nufi}' (score: {score})")
print()

# Check what phrases have "appelle"
print("Phrases with 'appelle':")
appelle_phrases = df[df['French'].str.lower().str.contains('appelle', regex=False, na=False)]
print(f"Found {len(appelle_phrases)} phrases")
for idx, row in appelle_phrases.head(10).iterrows():
    print(f"  - {row['French']}")

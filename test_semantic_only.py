"""
Test semantic search only for "je vais bien"
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

def semantic_search_only(text):
    """Pure semantic search - no exact or partial matching"""
    query_embedding = embedding_model.encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)

    k = 10  # Get top 10 results
    distances, indices = faiss_index.search(query_embedding, k)

    results = []
    for i in range(k):
        idx = indices[0][i]
        score = float(distances[0][i])
        score_percentage = score * 100

        result = df.iloc[idx].to_dict()
        result['match_score'] = score_percentage
        result['match_type'] = 'semantic'
        results.append(result)

    return results

# Test semantic search for "je vais bien"
print("\n" + "="*80)
print("SEMANTIC SEARCH RESULTS for 'je vais bien'")
print("="*80)

results = semantic_search_only("je vais bien")

for i, result in enumerate(results[:5], 1):  # Show top 5
    print(f"\n{i}. Score: {result['match_score']:.1f}%")
    print(f"   French: {result['French']}")
    print(f"   Nufi: {result['Nufi']}")

print(f"\nShowing top 5 of {len(results)} semantic matches")
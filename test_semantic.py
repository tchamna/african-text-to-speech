
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re

# Mock data based on user request
data = [
    {'French': 'Mon père est venu', 'Nufi': '...'},
    {'French': 'Mon enfant est malade', 'Nufi': '...'},
    {'French': 'Bonjour', 'Nufi': '...'},
    {'French': 'Comment ça va', 'Nufi': '...'}
]
df = pd.DataFrame(data)

# Load model
print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Create index
print("Creating index...")
embeddings = model.encode(df['French'].tolist())
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

def find_match(query):
    print(f"\nQuery: {query}")
    
    # 1. Exact/Keyword logic (simplified from app.py)
    query_lower = query.lower().strip()
    query_words = query_lower.split()
    
    # ... (skipping complex keyword logic for a moment to test raw semantic score first)
    
    # 2. Semantic Search
    query_vector = model.encode([query])
    k = 5
    D, I = index.search(np.array(query_vector).astype('float32'), k)
    
    print("Semantic Results:")
    for i in range(k):
        idx = I[0][i]
        if idx < len(df):
            score = 100 - (D[0][i] * 10) # Rough conversion
            print(f"  {i+1}. '{df.iloc[idx]['French']}' (Dist: {D[0][i]:.4f})")

find_match("Mon père est malade")

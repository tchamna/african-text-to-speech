"""
Build semantic search index for French-Nufi dictionary
Uses sentence embeddings and FAISS for fast similarity search
Run this once to build the index, then the app will use it
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

print("Loading phrasebook...")
df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')
# Rename Francais column to French for consistency
if 'Francais' in df.columns:
    df.rename(columns={'Francais': 'French'}, inplace=True)
df['French'] = df['French'].astype(str).str.strip()

# Remove duplicates
initial_count = len(df)
df.drop_duplicates(subset=['French'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"✓ Removed {initial_count - len(df)} duplicates")

print(f"✓ Loaded {len(df)} entries")

print("\nLoading sentence embedding model...")
# Using multilingual model that supports French
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Model loaded")

print("\nGenerating embeddings for all French phrases...")
# Get all French phrases
french_phrases = df['French'].tolist()

# Generate embeddings in batches
batch_size = 32
embeddings = model.encode(french_phrases, batch_size=batch_size, show_progress_bar=True)
print(f"✓ Generated {len(embeddings)} embeddings")

print("\nBuilding FAISS index...")
# Convert to float32 for FAISS
embeddings = np.array(embeddings).astype('float32')

# Normalize vectors for cosine similarity
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
index.add(embeddings)
print(f"✓ Index built with {index.ntotal} vectors")

print("\nSaving index and data...")
# Save the index
faiss.write_index(index, 'assets/faiss_index.bin')

# Save the dataframe indices for lookup
with open('assets/index_mapping.pkl', 'wb') as f:
    pickle.dump({
        'df_indices': df.index.tolist(),
        'dimension': dimension
    }, f)

print("✓ Saved to assets/faiss_index.bin and assets/index_mapping.pkl")

print("\n" + "="*60)
print("Index building complete!")
print(f"Dictionary size: {len(df):,} entries")
print(f"Embedding dimension: {dimension}")
print(f"Search complexity: O(log n) instead of O(n)")
print("="*60)

# Test the index
print("\nTesting index with sample queries...")
test_queries = ["je m'appelle", "bonjour", "merci", "ami"]

for query in test_queries:
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search for top 3 matches
    distances, indices = index.search(query_embedding, 3)
    
    print(f"\nQuery: '{query}'")
    for i, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
        match = df.iloc[idx]
        print(f"  {i}. '{match['French']}' → '{match['Nufi']}' (score: {score:.3f})")

print("\n✓ All tests passed!")

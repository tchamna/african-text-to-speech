import pandas as pd

df = pd.read_csv('assets/Nufi_Francais_Sentences_Phrases_From_Nufi_Tchamna_Dictionary_.csv')
df['French'] = df['French'].astype(str).str.strip()

# Search for sentences with "habite"
query = "J'habite à JONB"
query_lower = query.lower()

# Extract significant words
query_words = [w for w in query_lower.replace("'", " ").split() if len(w) >= 3]
print(f"Query: {query}")
print(f"Significant words: {query_words}")

# Find matches
for word in query_words[:3]:
    matches = df[df['French'].str.lower().str.contains(word, na=False, regex=False)]
    print(f"\nMatches containing '{word}' ({len(matches)} found):")
    for _, row in matches.head(5).iterrows():
        print(f"  - {row['French']} → {row['Nufi']}")

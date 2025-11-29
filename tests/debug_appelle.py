import pandas as pd

# Load the phrasebook
df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')
df.columns = ['French', 'Nufi']
df = df[df['French'].notna() & df['Nufi'].notna()]

text = "Je m'appelle Zona"
text_lower = text.lower().strip()
query_words = [w for w in text_lower.replace("'", " ").split() if len(w) >= 3]

print(f"Input: '{text}'")
print(f"Query words (≥3 chars): {query_words}")
print()

# Test multi-word matching
print("=== Testing multi-word combinations ===")
for i in range(len(query_words), 1, -1):
    for j in range(len(query_words) - i + 1):
        word_combo = query_words[j:j+i]
        print(f"\nTrying combination: {word_combo}")
        
        multi_matches = df.copy()
        for word in word_combo:
            multi_matches = multi_matches[multi_matches['French'].str.lower().str.contains(word, na=False, regex=False)]
        
        if not multi_matches.empty:
            print(f"  Found {len(multi_matches)} matches:")
            multi_matches['starts_with'] = multi_matches['French'].str.lower().str.startswith(tuple(word_combo))
            multi_matches['word_count'] = multi_matches['French'].str.split().str.len()
            multi_matches = multi_matches.sort_values(['starts_with', 'word_count'], ascending=[False, False])
            
            for idx, row in multi_matches.head(5).iterrows():
                print(f"    - {row['French']} (starts_with={row['starts_with']}, word_count={row['word_count']})")
        else:
            print("  No matches found")

print("\n=== Phrases with 'appelle' ===")
appelle_phrases = df[df['French'].str.lower().str.contains('appelle', na=False, regex=False)].head(10)
for idx, row in appelle_phrases.iterrows():
    french_lower = row['French'].lower()
    starts_with_je = french_lower.startswith('je')
    starts_with_appelle = french_lower.startswith('appelle')
    print(f"{row['French']}")
    print(f"  starts with 'je': {starts_with_je}, starts with 'appelle': {starts_with_appelle}")

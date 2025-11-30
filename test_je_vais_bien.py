import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load the phrasebook
df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')
df.columns = ['French', 'Nufi']
df = df[df['French'].notna() & df['Nufi'].notna()]
df = df[df['French'].str.strip().ne('') & df['Nufi'].str.strip().ne('')]
df['French'] = df['French'].astype(str).str.lower().str.strip()

# Load models
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
faiss_index = faiss.read_index('assets/faiss_index.bin')
with open('assets/index_mapping.pkl', 'rb') as f:
    index_data = pickle.load(f)

def find_closest_match(text):
    if df.empty:
        return None

    query_lower = text.lower().strip()

    # Step 1: Exact match
    exact_matches = df[df['French'].str.lower().str.strip() == query_lower]
    if not exact_matches.empty:
        result = exact_matches.iloc[0].to_dict()
        result['match_score'] = 100
        result['match_type'] = 'exact'
        print(f"Exact match: '{result['French']}' (score: 100)")
        return result

    # Step 2: Multi-word phrase match
    query_words_raw = query_lower.split()
    query_words = [w for w in query_words_raw if len(w) >= 2]

    if len(query_words) >= 2:
        for i in range(len(query_words), 1, -1):
            for j in range(len(query_words) - i + 1):
                word_combo = query_words[j:j+i]
                multi_matches = df.copy()
                for word in word_combo:
                    multi_matches = multi_matches[multi_matches['French'].str.lower().str.contains(word, na=False, regex=False)]

                if not multi_matches.empty:
                    multi_matches = multi_matches.copy()
                    first_word = word_combo[0]
                    multi_matches['starts_with_query'] = multi_matches['French'].str.lower().str.startswith(first_word)
                    multi_matches['length'] = multi_matches['French'].str.len()
                    multi_matches = multi_matches.sort_values(['starts_with_query', 'length'], ascending=[False, True])

                    result = multi_matches.iloc[0].to_dict()
                    result['match_score'] = 95 - (5 * (len(query_words) - i))
                    result['match_type'] = 'multi-word'
                    print(f"Multi-word match on {word_combo}: '{result['French']}' (score: {result['match_score']})")
                    return result

    # Step 3: Single-word partial match
    if query_words:
        significant_words = [w for w in query_words if len(w) >= 3]
        for word in significant_words[:3]:
            word_matches = df[df['French'].str.lower().str.contains(word, na=False, regex=False)]
            if not word_matches.empty:
                word_matches = word_matches.copy()
                word_matches['starts_with_word'] = word_matches['French'].str.lower().str.startswith(word)
                word_matches['length'] = word_matches['French'].str.len()
                word_matches = word_matches.sort_values(['starts_with_word', 'length'], ascending=[False, True])
                result = word_matches.iloc[0].to_dict()
                result['match_score'] = 85
                result['match_type'] = 'single-word'
                print(f"Single-word match on '{word}': '{result['French']}' (score: {result['match_score']})")
                return result

    return None

print('Testing: "Je vais bien"')
result = find_closest_match('Je vais bien')
if result:
    print(f"\nMatched French: {result['French']}")
    print(f"Nufi Translation: {result['Nufi']}")
    print(f"Score: {result['match_score']}")
    print(f"Type: {result['match_type']}")
else:
    print('No match found')
import pandas as pd

df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')

print("Testing phrases with 'la souris a faim' and similar:\n")

test_queries = [
    "la souris a faim",
    "la souris",
    "souris",
    "faim",
    "j'ai faim"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    matches = df[df['Francais'].str.contains(query, na=False, case=False)]
    print(f"Found {len(matches)} matches:")
    for _, row in matches.head(5).iterrows():
        print(f"  - {row['Francais']} → {row['Nufi']}")

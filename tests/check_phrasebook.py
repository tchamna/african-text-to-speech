import pandas as pd

df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')

print('Checking phrasebook quality...\n')
print(f'Total entries: {len(df)}')
print(f'Empty French: {df["Francais"].isna().sum()}')
print(f'Empty Nufi: {df["Nufi"].isna().sum()}')

print(f'\nVery long entries (>200 chars):')
long_entries = df[df["Francais"].str.len() > 200]
print(f'Count: {len(long_entries)}')
if len(long_entries) > 0:
    for i, row in long_entries.head(5).iterrows():
        print(f'  Row {i}: {row["Francais"][:100]}...')

print(f'\nChapter/section headers (likely not useful phrases):')
chapters = df[df["Francais"].str.contains('Chapitre|chapitre', na=False, case=False)]
print(f'Count: {len(chapters)}')

print(f'\nSample of actual useful phrases:')
useful = df[~df["Francais"].str.contains('Chapitre|chapitre', na=False, case=False)]
useful = useful[useful["Francais"].str.len() < 100]
print(f'Total useful phrases: {len(useful)}')
for i, row in useful.head(10).iterrows():
    print(f'  {row["Francais"]} → {row["Nufi"]}')

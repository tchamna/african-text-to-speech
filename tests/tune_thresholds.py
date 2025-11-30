"""
Simple tuning script to evaluate current thresholds on a small validation set.
Add more (query, expected_french) pairs to `validation_pairs` to expand coverage.

Run:
    & .\.venv\Scripts\Activate.ps1; python .\tests\tune_thresholds.py
"""
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import find_closest_match, find_top_semantic_matches

# Small validation set (query -> expected French match)
validation_pairs = [
    ("Je suis chanceux.", "Tu es chanceux."),
    ("Je suis fatigué.", "Je suis fatigué."),
    ("Bonne chance à toi!", "Bonne chance à toi!"),
    ("J'ai faim.", "J'ai faim"),
]

results = []
for q, expected in validation_pairs:
    match = find_closest_match(q)
    matched = match['French'] if match else None
    ok = (matched and expected.lower() in matched.lower())
    results.append((q, expected, matched, ok, match.get('match_score') if match else None))

print(json.dumps([{
    'query': r[0],
    'expected': r[1],
    'matched': r[2],
    'ok': r[3],
    'score': r[4]
} for r in results], ensure_ascii=False, indent=2))

# Print summary
good = sum(1 for r in results if r[3])
print(f"\nSummary: {good}/{len(results)} correct")

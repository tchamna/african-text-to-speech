import json
import sys
import os

# Ensure project root is on sys.path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import find_closest_match

q = "Je suis chanceux."
res = find_closest_match(q)
print(f"Query: {q}\n")
print(json.dumps(res, ensure_ascii=False, indent=2))

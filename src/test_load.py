import json
from pprint import pprint

file_path = "data/candidates.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    first_candidate = json.loads(next(f))

print("\nPROFILE:")
pprint(first_candidate["profile"])

print("\nSKILLS:")
pprint(first_candidate["skills"][:10])

print("\nCAREER HISTORY:")
pprint(first_candidate["career_history"][:2])

print("\nREDROB SIGNALS:")
pprint(first_candidate["redrob_signals"])
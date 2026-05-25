#!/usr/bin/env python
import json
from pathlib import Path
import sys

root = Path('experiments/results/rq1_security')
if len(sys.argv) > 1:
    root = Path(sys.argv[1])

res_file = root / 'rq1_results.json'
if not res_file.exists():
    print(f"Not found: {res_file}")
    sys.exit(1)

data = json.loads(res_file.read_text())
print("RQ1 Summary (attack x baseline -> final_acc)")
for item in data:
    atk = item.get('attack_type')
    base = item.get('baseline_method')
    acc = item.get('final_accuracy')
    print(f"- {atk} | {base}: final_acc={acc:.4f}")


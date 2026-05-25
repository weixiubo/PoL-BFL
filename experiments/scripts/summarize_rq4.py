#!/usr/bin/env python
import json
from pathlib import Path
import sys

root = Path('experiments/results/rq4_incentive')
if len(sys.argv) > 1:
    root = Path(sys.argv[1])

res_file = root / 'rq4_results.json'
if not res_file.exists():
    print(f"Not found: {res_file}")
    sys.exit(1)

data = json.loads(res_file.read_text())
print("RQ4 Incentive Summary (scenario -> avg_participation, attack_rate, final_acc)")
for r in data:
    print(f"- {r['scenario']}: part={r.get('avg_participation_rate', 0):.2f}, "
          f"attack_rate={r.get('avg_attack_success_rate', 0):.2f}, acc={r.get('final_accuracy', 0):.4f}")


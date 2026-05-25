#!/usr/bin/env python
import json
from pathlib import Path
import sys

root = Path('experiments/results/rq3_overhead')
if len(sys.argv) > 1:
    root = Path(sys.argv[1])

res_file = root / 'rq3_results.json'
if not res_file.exists():
    print(f"Not found: {res_file}")
    sys.exit(1)

data = json.loads(res_file.read_text())
print("RQ3 Overhead Summary")
for m in data:
    print(f"- {m['method']}: avg_round_time={m.get('avg_round_time', 0):.2f}s, "
          f"train={m.get('total_training_time', 0):.2f}s, comm={m.get('total_communication_mb', 0):.2f}MB")


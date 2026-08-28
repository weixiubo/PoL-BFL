#!/usr/bin/env python3
"""Execute the controlled single-threshold Kaizen-style Table 11 trial."""

from __future__ import annotations

import json

from experiments.final.run_cross_hardware_trial import parse_args, run_trial


if __name__ == "__main__":
    completed = run_trial(parse_args(), required_method="Kaizen")
    print(json.dumps(completed, indent=2, sort_keys=True))
    if completed["formal_accepted"] is not True:
        raise SystemExit(1)

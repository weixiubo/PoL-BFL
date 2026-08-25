from scripts.contract_gas_benchmark import build_evidence


def test_contract_gas_evidence_applies_paper_upper_bounds():
    raw = {
        "runtime_bytes": 22000,
        "commitment_gas": [84854, 84000],
        "receipt_gas": [110476],
        "slash_gas": "54013",
        "reward_claim_gas": "44702",
        "slashed_clients": 2,
    }
    targets = {
        "table_13_gas": {
            "commitment": 85000,
            "proof_receipt": 120000,
            "reward_claim": 45000,
            "slash": 65000,
            "honest_round_total": 225000,
        }
    }
    evidence = build_evidence(raw, targets)
    assert evidence["passed"]
    assert all(evidence["checks"].values())
    assert evidence["observed_gas"]["honest_round_total"] == 151651
    raw["reward_claim_gas"] = "45001"
    evidence = build_evidence(raw, targets)
    assert not evidence["passed"]
    assert not evidence["checks"]["reward_claim"]

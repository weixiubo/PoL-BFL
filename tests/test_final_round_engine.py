from decimal import Decimal

import numpy as np
import pytest

from polbfl.incentives import (
    EconomicParameters,
    IncentiveEngine,
    ParticipantAccount,
    ParticipantRole,
    ProofOutcome,
    ProtocolLedger,
)
from polbfl.protocol import PaperRoundEngine, RoundSubmission
from polbfl.sybil import TraceFingerprint


def _fingerprint(client, direction, indices):
    return TraceFingerprint(
        client,
        (client[0] if client[0] in "abcdef" else "f") * 64,
        ((0.0, 0.0), tuple(float(value) for value in direction)),
        tuple(indices),
    )


def _ledger(clients):
    ledger = ProtocolLedger(
        IncentiveEngine(
            EconomicParameters(
                base_reward=Decimal("1"),
                beta_work=Decimal("0"),
                beta_reputation=Decimal("0"),
                reputation_decay=Decimal("0.9"),
                slashing_ratio=Decimal("1"),
                challenge_probability=Decimal("0.2"),
                detection_probability=Decimal("0.965"),
                base_minimum_stake=Decimal("0.05"),
            )
        )
    )
    for client in clients:
        ledger.register(ParticipantAccount(client, ParticipantRole.CLIENT, Decimal("0.05")))
    return ledger


def test_round_engine_preserves_layer_order_and_slashing_boundary():
    honest = [
        ("a", 0.9, (1, 0)),
        ("b", 0.95, (0, 1)),
        ("c", 1.0, (-1, 0)),
        ("d", 1.05, (0, -1)),
        ("e", 1.1, (1, 1)),
    ]
    all_clients = [item[0] for item in honest] + ["sybil-1", "sybil-2", "reject"]
    ledger = _ledger(all_clients)
    submissions = [
        RoundSubmission(
            client,
            {"w": np.asarray([value], dtype=np.float64)},
            ProofOutcome.ACCEPT,
            Decimal("1"),
            _fingerprint(client, direction, (10 + index, 20 + index)),
        )
        for index, (client, value, direction) in enumerate(honest)
    ]
    submissions.extend(
        [
            RoundSubmission(
                "sybil-1",
                {"w": np.asarray([100.0])},
                ProofOutcome.ACCEPT,
                Decimal("1"),
                _fingerprint("sybil-1", (2, 3), (90, 91)),
            ),
            RoundSubmission(
                "sybil-2",
                {"w": np.asarray([100.0])},
                ProofOutcome.ACCEPT,
                Decimal("1"),
                _fingerprint("sybil-2", (2, 3), (90, 91)),
            ),
            RoundSubmission(
                "reject",
                {"w": np.asarray([-100.0])},
                ProofOutcome.REJECT,
                Decimal("0"),
                _fingerprint("reject", (-2, -3), (200, 201)),
            ),
        ]
    )
    result = PaperRoundEngine(
        ledger,
        aggregation_method="trimmed_mean",
        byzantine_bound=1,
    ).execute(round_id="round-engine", submissions=submissions)
    assert result.sybil_report.flagged_clients == frozenset({"sybil-1", "sybil-2"})
    assert result.aggregation.included_clients == ("a", "b", "c", "d", "e")
    assert float(result.aggregation.update["w"][0]) == 0.5
    assert result.settlement.eligible_clients == ("a", "b", "c", "d", "e")
    assert result.settlement.excluded_clients["reject"] == "proof_rejected"
    assert result.settlement.excluded_clients["sybil-1"] == "sybil_screened"
    assert ledger.accounts["reject"].stake == 0
    assert ledger.accounts["sybil-1"].stake == Decimal("0.05")
    assert len(result.execution_digest) == 64


def test_round_engine_ablation_disables_sybil_reputation_and_state_enforcement():
    clients = ["a", "b", "reject"]
    ledger = _ledger(clients)
    accounts_before = dict(ledger.accounts)
    submissions = [
        RoundSubmission(
            "a",
            {"w": np.asarray([1.0])},
            ProofOutcome.ACCEPT,
            Decimal("1"),
            _fingerprint("a", (2, 3), (90, 91)),
        ),
        RoundSubmission(
            "b",
            {"w": np.asarray([3.0])},
            ProofOutcome.ACCEPT,
            Decimal("1"),
            _fingerprint("b", (2, 3), (90, 91)),
        ),
        RoundSubmission(
            "reject",
            {"w": np.asarray([100.0])},
            ProofOutcome.REJECT,
            Decimal("0"),
            _fingerprint("reject", (-2, -3), (200, 201)),
        ),
    ]
    result = PaperRoundEngine(
        ledger,
        aggregation_method="trimmed_mean",
        byzantine_bound=0,
        enable_sybil_screening=False,
        enable_reputation_weighting=False,
        apply_economic_enforcement=False,
    ).execute(round_id="ablation", submissions=submissions)
    assert not result.sybil_report.flagged_clients
    assert result.aggregation.included_clients == ("a", "b")
    assert float(result.aggregation.update["w"][0]) == pytest.approx(2.0)
    assert result.settlement.excluded_clients == {
        "reject": "proof_rejected"
    }
    assert not result.settlement.slashed
    assert ledger.accounts == accounts_before
    assert "ablation" not in ledger.processed_rounds
    assert ledger.penalty_pool == 0


def test_round_engine_reduces_robust_budget_after_layer_one_prefilter():
    clients = [f"client-{index}" for index in range(6)]
    ledger = _ledger(clients)
    values = (0.0, 1.0, 2.0, 100.0, 101.0)
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1))
    submissions = [
        RoundSubmission(
            client,
            {"w": np.asarray([value], dtype=np.float64)},
            ProofOutcome.ACCEPT,
            Decimal("1"),
            _fingerprint(client, directions[index], (10 + index, 20 + index)),
        )
        for index, (client, value) in enumerate(zip(clients[:5], values))
    ]
    submissions.append(
        RoundSubmission(
            clients[-1],
            {"w": np.asarray([-1000.0], dtype=np.float64)},
            ProofOutcome.REJECT,
            Decimal("0"),
            _fingerprint(clients[-1], (9, -11), (99, 100)),
        )
    )
    result = PaperRoundEngine(
        ledger,
        aggregation_method="trimmed_mean",
        byzantine_bound=2,
    ).execute(round_id="residual-budget", submissions=submissions)
    # Initial reputation is 0.5. After one Layer-1 removal the residual f is
    # one, so the weighted extremes 0 and 50.5 are trimmed and 0.5, 1, 50 remain.
    assert float(result.aggregation.update["w"][0]) == pytest.approx(17.1666667)

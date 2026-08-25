from decimal import Decimal

import pytest

from polbfl.incentives import (
    ClientRoundOutcome,
    EconomicParameters,
    GasPriceQuote,
    IncentiveEngine,
    ParticipantAccount,
    ParticipantRole,
    ProofOutcome,
    ProtocolLedger,
    VerifierRoundOutcome,
)


def _ledger():
    engine = IncentiveEngine(
        EconomicParameters(
            base_reward=Decimal("1"),
            beta_work=Decimal("0.2"),
            beta_reputation=Decimal("0.3"),
            reputation_decay=Decimal("0.9"),
            slashing_ratio=Decimal("1"),
            challenge_probability=Decimal("0.2"),
            detection_probability=Decimal("0.965"),
            base_minimum_stake=Decimal("0.05"),
        )
    )
    ledger = ProtocolLedger(engine, verifier_reward=Decimal("0.1"))
    for client in ("honest", "reject", "timeout", "sybil", "byzantine", "not-audited"):
        ledger.register(
            ParticipantAccount(client, ParticipantRole.CLIENT, Decimal("0.05"))
        )
    for verifier in ("verifier-good", "verifier-bad"):
        ledger.register(
            ParticipantAccount(verifier, ParticipantRole.VERIFIER, Decimal("0.05"))
        )
    return ledger


def test_atomic_settlement_slashes_only_cryptographic_failures_and_verifier_misbehavior():
    ledger = _ledger()
    settlement = ledger.settle_round(
        round_id="round-1",
        client_outcomes=[
            ClientRoundOutcome("honest", ProofOutcome.ACCEPT, Decimal("1")),
            ClientRoundOutcome("reject", ProofOutcome.REJECT, Decimal("0")),
            ClientRoundOutcome("timeout", ProofOutcome.TIMEOUT, Decimal("0")),
            ClientRoundOutcome("sybil", ProofOutcome.ACCEPT, Decimal("1"), sybil_flagged=True),
            ClientRoundOutcome(
                "byzantine",
                ProofOutcome.ACCEPT,
                Decimal("1"),
                statistically_accepted=False,
            ),
            ClientRoundOutcome("not-audited", ProofOutcome.NOT_AUDITED, Decimal("0.5")),
        ],
        verifier_outcomes=[
            VerifierRoundOutcome("verifier-good", True, True),
            VerifierRoundOutcome("verifier-bad", True, False),
        ],
    )
    assert settlement.eligible_clients == ("honest", "not-audited")
    assert settlement.slashed == {
        "reject": Decimal("0.05"),
        "timeout": Decimal("0.05"),
        "verifier-bad": Decimal("0.05"),
    }
    assert ledger.penalty_pool == Decimal("0.15")
    assert ledger.accounts["reject"].stake == 0
    assert ledger.accounts["timeout"].stake == 0
    assert ledger.accounts["sybil"].stake == Decimal("0.05")
    assert ledger.accounts["byzantine"].stake == Decimal("0.05")
    assert settlement.rewards["honest"] == Decimal("1.35")
    assert settlement.rewards["sybil"] == 0
    assert settlement.rewards["verifier-good"] == Decimal("0.1")
    assert settlement.reputations["honest"] == Decimal("0.55")
    assert settlement.reputations["sybil"] == Decimal("0.45")
    assert len(settlement.settlement_digest) == 64

    with pytest.raises(ValueError, match="already"):
        ledger.settle_round(
            round_id="round-1",
            client_outcomes=[ClientRoundOutcome("honest", ProofOutcome.ACCEPT, Decimal("1"))],
        )


def test_authenticated_gas_quote_updates_responsive_minimum_stake():
    ledger = _ledger()
    quote = GasPriceQuote(
        provider_id="chainlink",
        observed_at_ns=10,
        gas_price_eth=Decimal("0.000001"),
        proof=b"oracle-proof",
    )
    with pytest.raises(ValueError, match="invalid"):
        ledger.update_minimum_client_stake(
            quote,
            operations_gas=Decimal("225000"),
            verify_quote=lambda _quote: False,
        )
    minimum = ledger.update_minimum_client_stake(
        quote,
        operations_gas=Decimal("225000"),
        verify_quote=lambda value: value.provider_id == "chainlink" and value.proof == b"oracle-proof",
    )
    assert minimum > Decimal("1")


def test_registration_enforces_client_and_verifier_collateral():
    ledger = _ledger()
    with pytest.raises(ValueError, match="below"):
        ledger.register(
            ParticipantAccount("underfunded", ParticipantRole.CLIENT, Decimal("0.049"))
        )

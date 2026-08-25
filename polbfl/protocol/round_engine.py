"""Ordered verify, Sybil-screen, robust-aggregate, and settle round engine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from polbfl.aggregation import (
    AggregationMethod,
    AggregationResult,
    VerifiedUpdate,
    aggregate_verified_updates,
)
from polbfl.crypto import canonical_json_bytes, domain_hash
from polbfl.incentives import (
    ClientRoundOutcome,
    ProofOutcome,
    ProtocolLedger,
    RoundSettlement,
    VerifierRoundOutcome,
)
from polbfl.sybil import SybilScreeningReport, TraceFingerprint, screen_trace_fingerprints


@dataclass(frozen=True)
class RoundSubmission:
    client_id: str
    update: Mapping[str, Any]
    proof_outcome: ProofOutcome
    normalized_work: Decimal
    fingerprint: TraceFingerprint
    statistically_accepted: bool = True

    def __post_init__(self) -> None:
        if self.client_id != self.fingerprint.client_id:
            raise ValueError("submission and trace fingerprint clients differ")


@dataclass(frozen=True)
class RoundExecutionResult:
    round_id: str
    sybil_report: SybilScreeningReport
    aggregation: AggregationResult
    settlement: RoundSettlement
    execution_digest: str


class PaperRoundEngine:
    def __init__(
        self,
        ledger: ProtocolLedger,
        *,
        aggregation_method: AggregationMethod | str = AggregationMethod.TRIMMED_MEAN,
        byzantine_bound: int = 0,
        sybil_cosine_threshold: float = 0.995,
        aggregation_device: str | None = None,
        enable_sybil_screening: bool = True,
        enable_reputation_weighting: bool = True,
        apply_economic_enforcement: bool = True,
    ):
        self.ledger = ledger
        self.aggregation_method = AggregationMethod(aggregation_method)
        self.byzantine_bound = int(byzantine_bound)
        self.sybil_cosine_threshold = float(sybil_cosine_threshold)
        self.aggregation_device = aggregation_device
        self.enable_sybil_screening = bool(enable_sybil_screening)
        self.enable_reputation_weighting = bool(enable_reputation_weighting)
        self.apply_economic_enforcement = bool(apply_economic_enforcement)

    def execute(
        self,
        *,
        round_id: str,
        submissions: Sequence[RoundSubmission],
        verifier_outcomes: Sequence[VerifierRoundOutcome] = (),
    ) -> RoundExecutionResult:
        if not submissions or len({item.client_id for item in submissions}) != len(submissions):
            raise ValueError("round submissions must be non-empty and client-unique")
        cryptographically_eligible = [
            item
            for item in submissions
            if item.proof_outcome in {ProofOutcome.ACCEPT, ProofOutcome.NOT_AUDITED}
        ]
        sybil_report = (
            screen_trace_fingerprints(
                [item.fingerprint for item in cryptographically_eligible],
                trajectory_cosine_threshold=self.sybil_cosine_threshold,
            )
            if self.enable_sybil_screening
            else SybilScreeningReport(frozenset(), tuple())
        )
        robust_inputs: list[VerifiedUpdate] = []
        for submission in submissions:
            account = self.ledger.accounts.get(submission.client_id)
            if account is None:
                raise ValueError(f"unregistered round client: {submission.client_id}")
            proof_eligible = submission.proof_outcome in {
                ProofOutcome.ACCEPT,
                ProofOutcome.NOT_AUDITED,
            }
            robust_inputs.append(
                VerifiedUpdate(
                    client_id=submission.client_id,
                    update=submission.update,
                    reputation=(
                        float(account.reputation)
                        if self.enable_reputation_weighting
                        else 1.0
                    ),
                    proof_eligible=proof_eligible and submission.statistically_accepted,
                    sybil_flagged=sybil_report.is_flagged(submission.client_id),
                )
            )
        prefiltered_count = sum(
            1
            for item in robust_inputs
            if not item.proof_eligible or item.sybil_flagged
        )
        eligible_count = len(robust_inputs) - prefiltered_count
        effective_byzantine_bound = min(
            max(0, self.byzantine_bound - prefiltered_count),
            max(0, (eligible_count - 1) // 2),
        )
        aggregation = aggregate_verified_updates(
            robust_inputs,
            method=self.aggregation_method,
            byzantine_bound=effective_byzantine_bound,
            device=self.aggregation_device,
        )
        accounts_before = dict(self.ledger.accounts)
        settlement = self.ledger.settle_round(
            round_id=round_id,
            client_outcomes=[
                ClientRoundOutcome(
                    client_id=item.client_id,
                    proof_outcome=item.proof_outcome,
                    normalized_work=item.normalized_work,
                    sybil_flagged=sybil_report.is_flagged(item.client_id),
                    statistically_accepted=item.statistically_accepted,
                )
                for item in submissions
            ],
            verifier_outcomes=verifier_outcomes,
        )
        if not self.apply_economic_enforcement:
            self.ledger.accounts = accounts_before
            self.ledger.processed_rounds.discard(round_id)
            self.ledger.penalty_pool -= sum(
                settlement.slashed.values(),
                Decimal("0"),
            )
            non_economic_body = {
                "round_id": round_id,
                "eligible_clients": list(settlement.eligible_clients),
                "excluded_clients": dict(settlement.excluded_clients),
                "preview_settlement_digest": settlement.settlement_digest,
            }
            settlement = RoundSettlement(
                round_id=round_id,
                eligible_clients=settlement.eligible_clients,
                excluded_clients=settlement.excluded_clients,
                slashed={},
                rewards={
                    submission.client_id: Decimal("0")
                    for submission in submissions
                },
                reputations={
                    submission.client_id: accounts_before[
                        submission.client_id
                    ].reputation
                    for submission in submissions
                },
                settlement_digest=domain_hash(
                    "POLBFL_NON_ECONOMIC_SETTLEMENT_V1",
                    canonical_json_bytes(non_economic_body),
                ),
            )
        body = {
            "round_id": round_id,
            "aggregation_method": aggregation.method.value,
            "aggregation_clients": list(aggregation.included_clients),
            "sybil_clients": sorted(sybil_report.flagged_clients),
            "settlement_digest": settlement.settlement_digest,
            "sybil_screening": self.enable_sybil_screening,
            "reputation_weighting": self.enable_reputation_weighting,
            "economic_enforcement": self.apply_economic_enforcement,
        }
        return RoundExecutionResult(
            round_id=round_id,
            sybil_report=sybil_report,
            aggregation=aggregation,
            settlement=settlement,
            execution_digest=domain_hash(
                "POLBFL_ROUND_EXECUTION_V1", canonical_json_bytes(body)
            ),
        )

"""Atomic stake, reputation, reward, timeout, and verifier settlement state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from typing import Callable, Iterable, Mapping

from polbfl.crypto import canonical_json_bytes, domain_hash
from polbfl.incentives.economics import IncentiveEngine, ONE, ZERO


class ParticipantRole(str, Enum):
    CLIENT = "client"
    VERIFIER = "verifier"


class ProofOutcome(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    TIMEOUT = "timeout"
    NOT_AUDITED = "not_audited"


@dataclass(frozen=True)
class ParticipantAccount:
    participant_id: str
    role: ParticipantRole
    stake: Decimal
    reputation: Decimal = Decimal("0.5")
    claimable_rewards: Decimal = ZERO
    active: bool = True

    def __post_init__(self) -> None:
        if not self.participant_id or self.stake < ZERO or self.claimable_rewards < ZERO:
            raise ValueError("participant identity and non-negative balances are required")
        if not ZERO <= self.reputation <= ONE:
            raise ValueError("participant reputation must be in [0, 1]")


@dataclass(frozen=True)
class ClientRoundOutcome:
    client_id: str
    proof_outcome: ProofOutcome
    normalized_work: Decimal
    sybil_flagged: bool = False
    statistically_accepted: bool = True

    def __post_init__(self) -> None:
        if not self.client_id or not ZERO <= self.normalized_work <= ONE:
            raise ValueError("client outcome requires an ID and normalized work")


@dataclass(frozen=True)
class VerifierRoundOutcome:
    verifier_id: str
    participated: bool
    correct_service: bool


@dataclass(frozen=True)
class GasPriceQuote:
    provider_id: str
    observed_at_ns: int
    gas_price_eth: Decimal
    proof: bytes

    def __post_init__(self) -> None:
        if not self.provider_id or self.observed_at_ns < 0 or self.gas_price_eth < ZERO or not self.proof:
            raise ValueError("authenticated non-negative gas quote is required")


@dataclass(frozen=True)
class RoundSettlement:
    round_id: str
    eligible_clients: tuple[str, ...]
    excluded_clients: Mapping[str, str]
    slashed: Mapping[str, Decimal]
    rewards: Mapping[str, Decimal]
    reputations: Mapping[str, Decimal]
    settlement_digest: str


class ProtocolLedger:
    def __init__(
        self,
        engine: IncentiveEngine,
        *,
        verifier_reward: Decimal = ZERO,
        minimum_verifier_stake: Decimal | None = None,
    ):
        if verifier_reward < ZERO:
            raise ValueError("verifier reward cannot be negative")
        self.engine = engine
        self.verifier_reward = verifier_reward
        self.minimum_client_stake = engine.parameters.base_minimum_stake
        self.minimum_verifier_stake = (
            engine.parameters.base_minimum_stake
            if minimum_verifier_stake is None
            else minimum_verifier_stake
        )
        if self.minimum_verifier_stake < ZERO:
            raise ValueError("minimum verifier stake cannot be negative")
        self.accounts: dict[str, ParticipantAccount] = {}
        self.processed_rounds: set[str] = set()
        self.penalty_pool = ZERO

    def register(self, account: ParticipantAccount) -> None:
        if account.participant_id in self.accounts:
            raise ValueError("participant is already registered")
        required = (
            self.minimum_client_stake
            if account.role == ParticipantRole.CLIENT
            else self.minimum_verifier_stake
        )
        if account.stake < required:
            raise ValueError("participant stake is below the active minimum")
        self.accounts[account.participant_id] = account

    def update_minimum_client_stake(
        self,
        quote: GasPriceQuote,
        *,
        operations_gas: Decimal,
        verify_quote: Callable[[GasPriceQuote], bool],
    ) -> Decimal:
        if operations_gas < ZERO or not verify_quote(quote):
            raise ValueError("gas-price quote is invalid")
        self.minimum_client_stake = self.engine.minimum_stake(
            gas_price=quote.gas_price_eth,
            operations_gas=operations_gas,
        )
        return self.minimum_client_stake

    def _account(self, participant_id: str, role: ParticipantRole) -> ParticipantAccount:
        account = self.accounts.get(participant_id)
        if account is None or account.role != role:
            raise ValueError(f"unknown {role.value}: {participant_id}")
        return account

    def settle_round(
        self,
        *,
        round_id: str,
        client_outcomes: Iterable[ClientRoundOutcome],
        verifier_outcomes: Iterable[VerifierRoundOutcome] = (),
    ) -> RoundSettlement:
        if not round_id or round_id in self.processed_rounds:
            raise ValueError("round settlement is empty or has already been processed")
        clients = tuple(client_outcomes)
        verifiers = tuple(verifier_outcomes)
        if not clients:
            raise ValueError("round settlement requires client outcomes")
        if len({item.client_id for item in clients}) != len(clients):
            raise ValueError("duplicate client outcome")
        if len({item.verifier_id for item in verifiers}) != len(verifiers):
            raise ValueError("duplicate verifier outcome")

        pending = dict(self.accounts)
        excluded: dict[str, str] = {}
        slashed: dict[str, Decimal] = {}
        rewards: dict[str, Decimal] = {}
        reputations: dict[str, Decimal] = {}
        eligible: list[str] = []

        for outcome in clients:
            account = self._account(outcome.client_id, ParticipantRole.CLIENT)
            if not account.active or account.stake < self.minimum_client_stake:
                raise ValueError(f"client is not stake-eligible: {outcome.client_id}")
            cryptographic_failure = outcome.proof_outcome in {
                ProofOutcome.REJECT,
                ProofOutcome.TIMEOUT,
            }
            if cryptographic_failure:
                remaining, penalty = self.engine.slash(account.stake)
                slashed[outcome.client_id] = penalty
                excluded[outcome.client_id] = (
                    "proof_rejected"
                    if outcome.proof_outcome == ProofOutcome.REJECT
                    else "audit_timeout"
                )
                success = False
                reward = ZERO
                account = replace(account, stake=remaining, active=remaining >= self.minimum_client_stake)
            elif outcome.sybil_flagged:
                excluded[outcome.client_id] = "sybil_screened"
                success = False
                reward = ZERO
            elif not outcome.statistically_accepted:
                excluded[outcome.client_id] = "robust_aggregation_rejected"
                success = False
                reward = ZERO
            else:
                success = True
                eligible.append(outcome.client_id)
                reward = self.engine.reward(
                    normalized_work=outcome.normalized_work,
                    reputation=account.reputation,
                )
            new_reputation = self.engine.update_reputation(
                current=account.reputation,
                verification_success=success,
            )
            account = replace(
                account,
                reputation=new_reputation,
                claimable_rewards=account.claimable_rewards + reward,
            )
            pending[outcome.client_id] = account
            rewards[outcome.client_id] = reward
            reputations[outcome.client_id] = new_reputation

        for outcome in verifiers:
            account = self._account(outcome.verifier_id, ParticipantRole.VERIFIER)
            if not account.active or account.stake < self.minimum_verifier_stake:
                raise ValueError(f"verifier is not stake-eligible: {outcome.verifier_id}")
            if outcome.participated and outcome.correct_service:
                reward = self.verifier_reward
                new_reputation = self.engine.update_reputation(
                    current=account.reputation,
                    verification_success=True,
                )
            elif outcome.participated:
                remaining, penalty = self.engine.slash(account.stake)
                slashed[outcome.verifier_id] = penalty
                account = replace(account, stake=remaining, active=remaining >= self.minimum_verifier_stake)
                reward = ZERO
                new_reputation = self.engine.update_reputation(
                    current=account.reputation,
                    verification_success=False,
                )
            else:
                reward = ZERO
                new_reputation = self.engine.update_reputation(
                    current=account.reputation,
                    verification_success=False,
                )
            pending[outcome.verifier_id] = replace(
                account,
                reputation=new_reputation,
                claimable_rewards=account.claimable_rewards + reward,
            )
            rewards[outcome.verifier_id] = reward
            reputations[outcome.verifier_id] = new_reputation

        body = {
            "round_id": round_id,
            "eligible_clients": sorted(eligible),
            "excluded_clients": excluded,
            "slashed": {key: str(value) for key, value in slashed.items()},
            "rewards": {key: str(value) for key, value in rewards.items()},
            "reputations": {key: str(value) for key, value in reputations.items()},
        }
        settlement = RoundSettlement(
            round_id=round_id,
            eligible_clients=tuple(sorted(eligible)),
            excluded_clients=dict(excluded),
            slashed=dict(slashed),
            rewards=dict(rewards),
            reputations=dict(reputations),
            settlement_digest=domain_hash(
                "POLBFL_ROUND_SETTLEMENT_V1", canonical_json_bytes(body)
            ),
        )
        self.accounts = pending
        self.penalty_pool += sum(slashed.values(), ZERO)
        self.processed_rounds.add(round_id)
        return settlement

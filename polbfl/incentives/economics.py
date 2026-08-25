"""Stake, reward, reputation, and rationality equations from the protocol."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(frozen=True)
class EconomicParameters:
    base_reward: Decimal
    beta_work: Decimal
    beta_reputation: Decimal
    reputation_decay: Decimal
    slashing_ratio: Decimal
    challenge_probability: Decimal
    detection_probability: Decimal
    base_minimum_stake: Decimal
    stake_margin: Decimal = ZERO

    def __post_init__(self) -> None:
        probabilities = (
            self.reputation_decay,
            self.slashing_ratio,
            self.challenge_probability,
            self.detection_probability,
        )
        if any(value < ZERO or value > ONE for value in probabilities):
            raise ValueError("probability-like economic parameters must be in [0, 1]")
        if self.base_reward < ZERO or self.base_minimum_stake < ZERO or self.stake_margin < ZERO:
            raise ValueError("reward, stake, and margin must be non-negative")
        if self.slashing_ratio == ZERO:
            raise ValueError("slashing ratio must be positive")


class IncentiveEngine:
    def __init__(self, parameters: EconomicParameters):
        self.parameters = parameters

    @property
    def per_round_detection_probability(self) -> Decimal:
        return self.parameters.challenge_probability * self.parameters.detection_probability

    def minimum_stake(self, *, gas_price: Decimal, operations_gas: Decimal) -> Decimal:
        denominator = (
            self.parameters.slashing_ratio
            * self.parameters.challenge_probability
            * self.parameters.detection_probability
        )
        if denominator <= ZERO:
            raise ValueError("positive challenge and detection probabilities are required")
        responsive = gas_price * operations_gas / denominator + self.parameters.stake_margin
        return max(self.parameters.base_minimum_stake, responsive)

    def reward(self, *, normalized_work: Decimal, reputation: Decimal) -> Decimal:
        if not (ZERO <= normalized_work <= ONE and ZERO <= reputation <= ONE):
            raise ValueError("work and reputation must be normalized to [0, 1]")
        return self.parameters.base_reward * (
            ONE
            + self.parameters.beta_work * normalized_work
            + self.parameters.beta_reputation * reputation
        )

    def update_reputation(self, *, current: Decimal, verification_success: bool) -> Decimal:
        if not ZERO <= current <= ONE:
            raise ValueError("reputation must be in [0, 1]")
        indicator = ONE if verification_success else ZERO
        alpha = self.parameters.reputation_decay
        return alpha * current + (ONE - alpha) * indicator

    def slash(self, stake: Decimal) -> tuple[Decimal, Decimal]:
        if stake < ZERO:
            raise ValueError("stake must be non-negative")
        penalty = stake * self.parameters.slashing_ratio
        return stake - penalty, penalty

    def honest_dominates(self, *, stake: Decimal, expected_reward: Decimal, saved_cost: Decimal) -> bool:
        if min(stake, expected_reward, saved_cost) < ZERO:
            raise ValueError("utility inputs must be non-negative")
        expected_detection_loss = self.per_round_detection_probability * (
            expected_reward + self.parameters.slashing_ratio * stake
        )
        return expected_detection_loss > saved_cost

    @staticmethod
    def participation_is_rational(*, expected_reward: Decimal, honest_cost: Decimal) -> bool:
        return expected_reward - honest_cost >= ZERO

"""Factory for server-side aggregation implementations."""

from .balance import balanceAggregator
from .fedavg import fedavgAggregator
from .fools_gold import FoolsGoldAggregator
from .krum import krumAggregator
from .median import medianAggregator
from .shapley_fl import ShapleyFLAggregator


_AGGREGATORS = {
    "balance": balanceAggregator,
    "fedavg": fedavgAggregator,
    "foolsgold": FoolsGoldAggregator,
    "krum": krumAggregator,
    "median": medianAggregator,
    "shapleyfl": ShapleyFLAggregator,
}


def create_aggregator(name: str, **kwargs):
    """Create an aggregator by a case- and punctuation-insensitive name."""
    normalized = "".join(character for character in name.lower() if character.isalnum())
    try:
        aggregator_type = _AGGREGATORS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown aggregator {name!r}; expected one of {sorted(_AGGREGATORS)}"
        ) from exc
    return aggregator_type(**kwargs)


def available_aggregators():
    return tuple(sorted(_AGGREGATORS))

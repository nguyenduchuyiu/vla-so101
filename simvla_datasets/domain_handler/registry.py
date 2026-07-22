from __future__ import annotations
from typing import Dict, Type
from .base import DomainHandler
from .cf_balanced import CFBalancedHandler
from .so101_balanced_counterfactual import SO101BalancedCounterfactualHandler

# Registry for dataset handlers
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    "so101_balanced_counterfactual": SO101BalancedCounterfactualHandler,
    "cf_balanced": CFBalancedHandler,
}


def get_handler_cls(dataset_name: str) -> Type[DomainHandler]:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Available: {sorted(_REGISTRY)}."
        )
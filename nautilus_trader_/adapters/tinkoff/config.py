"""
Tinkoff configuration.
"""

from dataclasses import dataclass
from typing import Optional

from nautilus_trader.core.correctness import PyCondition


@dataclass
class TinkoffDataClientConfig:
    """
    Configuration for Tinkoff live data client.
    """
    api_key: str
    account_id: str
    sandbox_mode: bool = False
    max_subscriptions: int = 100
    max_requests_per_minute: int = 180

    def __post_init__(self):
        PyCondition.not_empty(self.api_key, "api_key")
        PyCondition.not_empty(self.account_id, "account_id")
        PyCondition.positive(self.max_subscriptions, "max_subscriptions")
        PyCondition.positive(self.max_requests_per_minute, "max_requests_per_minute")


@dataclass
class TinkoffExecClientConfig:
    """
    Configuration for Tinkoff live execution client.
    """
    api_key: str
    account_id: str
    sandbox_mode: bool = False
    max_requests_per_minute: int = 180

    def __post_init__(self):
        PyCondition.not_empty(self.api_key, "api_key")
        PyCondition.not_empty(self.account_id, "account_id")
        PyCondition.positive(self.max_requests_per_minute, "max_requests_per_minute")
# t_adapter/config.py

# -*- coding: utf-8 -*-
"""
Tinkoff Invest adapter configuration.
"""

from typing import Optional, List

from nautilus_trader.config import LiveDataClientConfig, LiveExecClientConfig

from t_adapter.enums import TinkoffProductType


class TAdapterDataClientConfig(LiveDataClientConfig, frozen=True):
    """
    Configuration for ``TAdapterDataClient`` instances.
    """

    api_token: str
    """The Tinkoff Invest API token."""

    product_types: Optional[List[TinkoffProductType]] = None
    """The product types to load."""

    is_sandbox: bool = False
    """If the client is connecting to the Tinkoff Invest sandbox API."""


class TAdapterExecClientConfig(LiveExecClientConfig, frozen=True):
    """
    Configuration for ``TAdapterExecutionClient`` instances.
    """

    api_token: str
    """The Tinkoff Invest API token."""

    account_id: Optional[str] = None
    """The specific account ID to use. If None, the default account will be used."""

    is_sandbox: bool = False
    """If the client is connecting to the Tinkoff Invest sandbox API."""

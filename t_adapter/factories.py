# t_adapter/factories.py

# -*- coding: utf-8 -*-
"""
Tinkoff Invest adapter factories.
"""

from typing import Optional

from nautilus_trader.adapters.t_adapter.config import TAdapterDataClientConfig, TAdapterExecClientConfig
from nautilus_trader.adapters.t_adapter.data_client import TAdapterDataClient
from nautilus_trader.adapters.t_adapter.execution_client import TAdapterExecutionClient
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.model.identifiers import AccountId


class TAdapterLiveDataClientFactory(LiveDataClientFactory):
    """
    Factory for creating Tinkoff Invest data clients.
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config: TAdapterDataClientConfig,
        msgbus,
        cache,
        clock,
        logger,
        **kwargs,
    ) -> TAdapterDataClient:
        """
        Create a new Tinkoff Invest data client.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        name : str
            The name for the client.
        config : TAdapterDataClientConfig
            The configuration for the client.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.
        logger : Logger
            The logger for the client.

        Returns
        -------
        TAdapterDataClient

        """
        # Import here to avoid circular imports if needed
        from tinkoff.invest import Client
        from t_adapter.instrument_provider import TAdapterInstrumentProvider

        # Initialize the Tinkoff Invest client
        # The target (sandbox/production) should be set based on config.is_sandbox
        # This assumes the Client can be initialized like this. Check invest-python docs.
        target = "https://sandbox-invest-public-api.tinkoff.ru:443" if config.is_sandbox else "https://invest-public-api.tinkoff.ru:443"
        client = Client(config.api_token, target=target)

        # Initialize instrument provider
        instrument_provider = TAdapterInstrumentProvider(client=client, config=config, logger=logger)

        # Return the data client
        return TAdapterDataClient(
            loop=loop,
            client=client,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
            name=name,
        )


class TAdapterLiveExecClientFactory(LiveExecClientFactory):
    """
    Factory for creating Tinkoff Invest execution clients.
    """

    @staticmethod
    def create(
        loop,
        name: str,
        config: TAdapterExecClientConfig,
        msgbus,
        cache,
        clock,
        logger,
        **kwargs,
    ) -> TAdapterExecutionClient:
        """
        Create a new Tinkoff Invest execution client.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        name : str
            The name for the client.
        config : TAdapterExecClientConfig
            The configuration for the client.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.
        logger : Logger
            The logger for the client.

        Returns
        -------
        TAdapterExecutionClient

        """
        from tinkoff.invest import Client
        from t_adapter.instrument_provider import TAdapterInstrumentProvider

        # Initialize the Tinkoff Invest client
        target = "https://sandbox-invest-public-api.tinkoff.ru:443" if config.is_sandbox else "https://invest-public-api.tinkoff.ru:443"
        client = Client(config.api_token, target=target)

        # Account ID
        account_id = AccountId(f"{name}-{config.account_id or 'default'}") # Create AccountId

        # Initialize instrument provider (could be shared with data client)
        instrument_provider = TAdapterInstrumentProvider(client=client, config=config, logger=logger) # Config might need adjustment

        # Return the execution client
        return TAdapterExecutionClient(
            loop=loop,
            client=client,
            account_id=account_id,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
            name=name,
        )

"""
Tinkoff client factories.
"""

from nautilus_trader.adapters.tinkoff.config import TinkoffDataClientConfig
from nautilus_trader.adapters.tinkoff.config import TinkoffExecClientConfig
from nautilus_trader.adapters.tinkoff.data import TinkoffLiveDataClient
from nautilus_trader.adapters.tinkoff.execution import TinkoffLiveExecutionClient
from nautilus_trader.adapters.tinkoff.instruments import TinkoffInstrumentProvider
from nautilus_trader.adapters.tinkoff.client import TinkoffClient
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.common.uuid import UUID4
from nautilus_trader.msgbus.bus import MessageBus


def TinkoffLiveDataClientFactory(
        name: str,
        config: dict,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        logger: Logger,
):
    """
    Create a new Tinkoff live data client.
    """
    # Создаем конфигурацию
    data_config = TinkoffDataClientConfig(**config)

    # Создаем Tinkoff клиент
    tinkoff_client = TinkoffClient(
        clock=clock,
        logger=logger,
        data_config=data_config,
    )

    # Создаем провайдер инструментов
    instrument_provider = TinkoffInstrumentProvider(
        client=tinkoff_client,
        logger=logger,
        clock=clock,
        cache=cache,
        data_config=data_config,
    )

    # Создаем клиент данных
    return TinkoffLiveDataClient(
        loop=clock.loop,
        client=tinkoff_client,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        logger=logger,
        instrument_provider=instrument_provider,
        data_config=data_config,
    )


def TinkoffLiveExecClientFactory(
        name: str,
        config: dict,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        logger: Logger,
):
    """
    Create a new Tinkoff live execution client.
    """
    # Создаем конфигурацию
    exec_config = TinkoffExecClientConfig(**config)

    # Создаем Tinkoff клиент
    tinkoff_client = TinkoffClient(
        clock=clock,
        logger=logger,
        exec_config=exec_config,
    )

    # Создаем провайдер инструментов
    instrument_provider = TinkoffInstrumentProvider(
        client=tinkoff_client,
        logger=logger,
        clock=clock,
        cache=cache,
        exec_config=exec_config,
    )

    # Создаем клиент исполнения
    return TinkoffLiveExecutionClient(
        loop=clock.loop,
        client=tinkoff_client,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        logger=logger,
        instrument_provider=instrument_provider,
        exec_config=exec_config,
    )
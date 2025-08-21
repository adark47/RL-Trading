"""
Tinkoff instrument provider.
"""

import asyncio
from typing import Dict, List, Optional

from nautilus_trader.adapters.tinkoff.client import TinkoffClient
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.common.uuid import UUID4
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import InstrumentClass
from nautilus_trader.model.enums import InstrumentStatus
from nautilus_trader.model.enums import VenueType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from nautilus_trader.adapters.tinkoff.common import TINKOFF_VENUE
from nautilus_trader.adapters.tinkoff.config import TinkoffDataClientConfig
from nautilus_trader.adapters.tinkoff.config import TinkoffExecClientConfig
from nautilus_trader.adapters.tinkoff.data import tinkoff_instrument_to_nautilus
from nautilus_trader.adapters.tinkoff.schemas import TinkoffInstrument


class TinkoffInstrumentProvider:
    """
    Provides a means of loading instruments for the Tinkoff adapter.
    """

    def __init__(
            self,
            client: TinkoffClient,
            logger: Logger,
            clock: LiveClock,
            cache: Cache,
            data_config: Optional[TinkoffDataClientConfig] = None,
            exec_config: Optional[TinkoffExecClientConfig] = None,
    ):
        self._client = client
        self._log = logger
        self._clock = clock
        self._cache = cache
        self._data_config = data_config
        self._exec_config = exec_config

        self._instruments: Dict[InstrumentId, Instrument] = {}
        self._instrument_ids: Dict[str, InstrumentId] = {}
        self._loaded = False

    @property
    def venue(self) -> Venue:
        """Return the venue for this provider."""
        return TINKOFF_VENUE

    @property
    def instruments(self) -> List[Instrument]:
        """Return all loaded instruments."""
        return list(self._instruments.values())

    def find(self, instrument_id: InstrumentId) -> Optional[Instrument]:
        """Find an instrument by its ID."""
        return self._instruments.get(instrument_id)

    async def load_all(self, reload: bool = False) -> None:
        """
        Load all instruments from Tinkoff.
        """
        if self._loaded and not reload:
            return

        self._log.info("Loading instruments from Tinkoff...")

        # Загружаем инструменты через TinkoffPy
        instruments = []

        # Акции
        instruments.extend(self._client._provider.get_all_shares())
        # Облигации
        instruments.extend(self._client._provider.get_all_bonds())
        # Фьючерсы
        instruments.extend(self._client._provider.get_all_futures())
        # Валюты
        instruments.extend(self._client._provider.get_all_currencies())

        # Преобразуем в формат Nautilus Trader
        for tinkoff_instrument in instruments:
            nautilus_instrument = tinkoff_instrument_to_nautilus(tinkoff_instrument)
            self._instruments[nautilus_instrument.id] = nautilus_instrument
            self._instrument_ids[tinkoff_instrument.figi] = nautilus_instrument.id

        self._loaded = True
        self._log.info(f"Loaded {len(self._instruments)} instruments from Tinkoff")

    def get_cached_instrument(self, instrument_id: InstrumentId) -> Optional[Instrument]:
        """
        Get a cached instrument by its ID.
        """
        return self.find(instrument_id)

    def get_cached_instrument_id(self, figi: str) -> Optional[InstrumentId]:
        """
        Get a cached instrument ID by its FIGI.
        """
        return self._instrument_ids.get(figi)
"""
Tinkoff data client.
"""

import asyncio
from typing import Any, Dict, List, Optional

from nautilus_trader.adapters.tinkoff.client import TinkoffClient
from nautilus_trader.adapters.tinkoff.instruments import TinkoffInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.common.uuid import UUID4
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.core.uuid import uuid4
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import BookLevel
from nautilus_trader.model.enums import InstrumentClass
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.enums import RecordFlag
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.msgbus.bus import MessageBus

from nautilus_trader.adapters.tinkoff.common import TINKOFF_VENUE
from nautilus_trader.adapters.tinkoff.config import TinkoffDataClientConfig
from nautilus_trader.adapters.tinkoff.schemas import TinkoffBar
from nautilus_trader.adapters.tinkoff.schemas import TinkoffOrderBook
from nautilus_trader.adapters.tinkoff.schemas import TinkoffTrade


class TinkoffLiveDataClient:
    """
    Tinkoff client for live market data.
    """

    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            client: TinkoffClient,
            msgbus: MessageBus,
            cache: Cache,
            clock: LiveClock,
            logger: Logger,
            instrument_provider: TinkoffInstrumentProvider,
            data_config: TinkoffDataClientConfig,
    ):
        self._loop = loop
        self._client = client
        self._msgbus = msgbus
        self._cache = cache
        self._clock = clock
        self._log = logger
        self._instrument_provider = instrument_provider
        self._config = data_config

        self._client_id = ClientId(f"TINKOFF-{id(self)}")
        self._subscribed = set()

        # Устанавливаем обработчики событий
        self._client._event_handlers["on_candle"] = self._on_candle
        self._client._event_handlers["on_trade"] = self._on_trade
        self._client._event_handlers["on_order_book"] = self._on_order_book

    async def connect(self):
        """Connect to the Tinkoff API."""
        self._log.info("Connecting to Tinkoff data API...")
        await self._client.connect()
        self._log.info("Connected to Tinkoff data API")

        # Загружаем инструменты
        await self._instrument_provider.load_all()

    async def disconnect(self):
        """Disconnect from the Tinkoff API."""
        self._log.info("Disconnecting from Tinkoff data API...")
        await self._client.disconnect()
        self._log.info("Disconnected from Tinkoff data API")

    def subscribe(self, data_type: DataType):
        """Subscribe to data type."""
        # Для Tinkoff мы используем подписки через BarType
        if not isinstance(data_type.data, BarType):
            self._log.error(f"Unsupported data type for subscription: {data_type}")
            return

        bar_type = data_type.data
        instrument = self._cache.instrument(bar_type.instrument_id)
        if not instrument:
            self._log.error(f"Cannot subscribe to {bar_type}: instrument not found")
            return

        # Сохраняем подписку
        self._subscribed.add(bar_type)

        # Подписываемся на свечи
        self._client.subscribe_candles(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            bar_type.spec.aggregation.value
        )

    def unsubscribe(self, data_type: DataType):
        """Unsubscribe from data type."""
        if not isinstance(data_type.data, BarType):
            self._log.error(f"Unsupported data type for unsubscription: {data_type}")
            return

        bar_type = data_type.data
        if bar_type in self._subscribed:
            self._subscribed.remove(bar_type)
            # TODO: Реализовать отписку от конкретных свечей

    def _on_candle(self, candle):
        """Обработка свечи от Tinkoff."""
        # Преобразуем свечу Tinkoff в формат Nautilus
        bar = self._candle_to_bar(candle)
        if not bar:
            return

        # Публикуем свечу
        self._handle_data(bar)

    def _on_trade(self, trade):
        """Обработка сделки от Tinkoff."""
        # Преобразуем сделку Tinkoff в формат Nautilus
        trade_tick = self._trade_to_trade_tick(trade)
        if not trade_tick:
            return

        # Публикуем сделку
        self._handle_data(trade_tick)

    def _on_order_book(self, order_book):
        """Обработка стакана от Tinkoff."""
        # Преобразуем стакан Tinkoff в формат Nautilus
        quote_tick = self._order_book_to_quote_tick(order_book)
        if not quote_tick:
            return

        # Публикуем котировку
        self._handle_data(quote_tick)

    def _candle_to_bar(self, candle) -> Optional[Bar]:
        """Преобразует свечу Tinkoff в Bar Nautilus."""
        # Получаем инструмент по FIGI
        instrument_id = self._instrument_provider.get_cached_instrument_id(candle.figi)
        if not instrument_id:
            return None

        instrument = self._cache.instrument(instrument_id)
        if not instrument:
            return None

        # Определяем таймфрейм
        tf = self._client._provider.tinkoff_timeframe_to_timeframe(candle.interval)[0]

        # Создаем BarType
        bar_type = BarType(
            instrument_id=instrument_id,
            aggregation=BarAggregation(tf[0].upper()),
            price_type=PriceType.LAST,
            step=1
        )

        # Преобразуем время
        dt_utc = self._client._provider.utc_to_msk_datetime(candle.time)
        ts_init = dt_to_unix_nanos(dt_utc)

        # Преобразуем цены
        open_price = self._client._provider.tinkoff_price_to_price(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            candle.open
        )
        high_price = self._client._provider.tinkoff_price_to_price(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            candle.high
        )
        low_price = self._client._provider.tinkoff_price_to_price(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            candle.low
        )
        close_price = self._client._provider.tinkoff_price_to_price(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            candle.close
        )

        # Создаем бар
        return Bar(
            bar_type=bar_type,
            open=Price(open_price, instrument.price_precision),
            high=Price(high_price, instrument.price_precision),
            low=Price(low_price, instrument.price_precision),
            close=Price(close_price, instrument.price_precision),
            volume=Quantity(candle.volume, instrument.quantity_precision),
            ts_event=ts_init,
            ts_init=ts_init,
        )

    def _trade_to_trade_tick(self, trade) -> Optional[TradeTick]:
        """Преобразует сделку Tinkoff в TradeTick Nautilus."""
        # Получаем инструмент по FIGI
        instrument_id = self._instrument_provider.get_cached_instrument_id(trade.figi)
        if not instrument_id:
            return None

        instrument = self._cache.instrument(instrument_id)
        if not instrument:
            return None

        # Преобразуем время
        dt_utc = self._client._provider.utc_to_msk_datetime(trade.time)
        ts_event = dt_to_unix_nanos(dt_utc)

        # Преобразуем цену
        price = self._client._provider.tinkoff_price_to_price(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            trade.price
        )

        # Определяем сторону агрессора
        aggressor = AggressorSide.BUY if trade.direction == "BUY" else AggressorSide.SELL

        # Создаем TradeTick
        return TradeTick(
            instrument_id=instrument_id,
            price=Price(price, instrument.price_precision),
            size=Quantity(trade.quantity, instrument.quantity_precision),
            aggressor=aggressor,
            trade_id=trade.id,
            ts_event=ts_event,
            ts_init=ts_event,
        )

    def _order_book_to_quote_tick(self, order_book) -> Optional[QuoteTick]:
        """Преобразует стакан Tinkoff в QuoteTick Nautilus."""
        # Получаем инструмент по FIGI
        instrument_id = self._instrument_provider.get_cached_instrument_id(order_book.figi)
        if not instrument_id:
            return None

        instrument = self._cache.instrument(instrument_id)
        if not instrument:
            return None

        # Преобразуем время
        dt_utc = self._client._provider.utc_to_msk_datetime(order_book.time)
        ts_event = dt_to_unix_nanos(dt_utc)

        # Получаем лучшие цены
        bid_price = self._client._provider.tinkoff_price_to_price(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            order_book.bids[0].price if order_book.bids else 0
        )
        ask_price = self._client._provider.tinkoff_price_to_price(
            instrument.raw_symbol.value,
            instrument.symbol.value,
            order_book.asks[0].price if order_book.asks else 0
        )

        bid_size = order_book.bids[0].quantity if order_book.bids else 0
        ask_size = order_book.asks[0].quantity if order_book.asks else 0

        # Создаем QuoteTick
        return QuoteTick(
            instrument_id=instrument_id,
            bid_price=Price(bid_price, instrument.price_precision),
            ask_price=Price(ask_price, instrument.price_precision),
            bid_size=Quantity(bid_size, instrument.quantity_precision),
            ask_size=Quantity(ask_size, instrument.quantity_precision),
            ts_event=ts_event,
            ts_init=ts_event,
        )

    def _handle_data(self, data):
        """Публикует данные через шину сообщений."""
        self._msgbus.publish(
            topic=f"data.{type(data).__name__.lower()}",
            data=data,
        )
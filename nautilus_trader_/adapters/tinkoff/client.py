"""
Tinkoff API client wrapper.
"""

import asyncio
import logging
from typing import Dict, Optional

from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from TinkoffPy import TinkoffPy

from nautilus_trader.adapters.tinkoff.common import TINKOFF_VENUE
from nautilus_trader.adapters.tinkoff.config import TinkoffDataClientConfig
from nautilus_trader.adapters.tinkoff.config import TinkoffExecClientConfig


class TinkoffClient:
    """
    Client for interacting with the Tinkoff Invest API.
    """

    def __init__(
            self,
            clock: LiveClock,
            logger: Logger,
            data_config: Optional[TinkoffDataClientConfig] = None,
            exec_config: Optional[TinkoffExecClientConfig] = None,
    ):
        """
        Initialize a new Tinkoff client.

        Parameters
        ----------
        clock : LiveClock
            The clock for the client.
        logger : Logger
            The logger for the client.
        data_config : TinkoffDataClientConfig, optional
            The data client configuration.
        exec_config : TinkoffExecClientConfig, optional
            The execution client configuration.
        """
        PyCondition.not_none(clock, "clock")
        PyCondition.not_none(logger, "logger")

        self._clock = clock
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Создаем экземпляр TinkoffPy
        self._provider = TinkoffPy()

        # Словарь для хранения обработчиков событий
        self._event_handlers = {}

        # Флаг подключения
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """
        Return whether the client is connected.
        """
        return self._is_connected

    async def connect(self):
        """
        Connect to the Tinkoff Invest API.
        """
        if self._is_connected:
            return

        self._log.info("Connecting to Tinkoff Invest API...")

        # Устанавливаем обработчики событий
        self._setup_event_handlers()

        # Запускаем потоки подписок
        self._start_subscription_threads()

        self._is_connected = True
        self._log.info("Connected to Tinkoff Invest API")

    async def disconnect(self):
        """
        Disconnect from the Tinkoff Invest API.
        """
        if not self._is_connected:
            return

        self._log.info("Disconnecting from Tinkoff Invest API...")

        # Останавливаем потоки подписок
        self._stop_subscription_threads()

        # Закрываем соединение
        self._provider.close_channel()

        self._is_connected = False
        self._log.info("Disconnected from Tinkoff Invest API")

    def _setup_event_handlers(self):
        """Настраиваем обработчики событий от TinkoffPy"""
        # Свечи
        self._provider.on_candle = self._handle_candle
        # Сделки
        self._provider.on_trade = self._handle_trade
        # Стакан
        self._provider.on_order_book = self._handle_order_book
        # Последняя цена
        self._provider.on_last_price = self._handle_last_price
        # Статус торгов
        self._provider.on_trading_status = self._handle_trading_status
        # Портфель
        self._provider.on_portfolio = self._handle_portfolio
        # Позиции
        self._provider.on_position = self._handle_position
        # Сделки по заявкам
        self._provider.on_order_trades = self._handle_order_trades

    def _start_subscription_threads(self):
        """Запускаем потоки подписок на рыночные данные"""
        # Запускаем поток для обработки рыночных данных
        self._marketdata_thread = Thread(target=self._provider.subscriptions_marketdata_handler, daemon=True)
        self._marketdata_thread.start()

        # Запускаем поток для обработки портфеля
        self._portfolio_thread = Thread(target=self._provider.subscriptions_portfolio_handler, args=(self.account_id,),
                                        daemon=True)
        self._portfolio_thread.start()

        # Запускаем поток для обработки позиций
        self._positions_thread = Thread(target=self._provider.subscriptions_positions_handler, args=(self.account_id,),
                                        daemon=True)
        self._positions_thread.start()

        # Запускаем поток для обработки сделок
        self._trades_thread = Thread(target=self._provider.subscriptions_trades_handler, args=([self.account_id],),
                                     daemon=True)
        self._trades_thread.start()

    def _stop_subscription_threads(self):
        """Останавливаем потоки подписок"""
        # Останавливаем обработку рыночных данных
        self._provider.subscription_marketdata_queue.put(None)
        self._marketdata_thread.join(timeout=1.0)

        # Останавливаем обработку портфеля
        self._provider.subscription_portfolio_queue.put(None)
        self._portfolio_thread.join(timeout=1.0)

        # Останавливаем обработку позиций
        self._provider.subscription_positions_queue.put(None)
        self._positions_thread.join(timeout=1.0)

        # Останавливаем обработку сделок
        self._provider.subscription_trades_queue.put(None)
        self._trades_thread.join(timeout=1.0)

    # Обработчики событий
    def _handle_candle(self, candle):
        """Обработка свечи"""
        if "on_candle" in self._event_handlers:
            self._event_handlers["on_candle"](candle)

    def _handle_trade(self, trade):
        """Обработка сделки"""
        if "on_trade" in self._event_handlers:
            self._event_handlers["on_trade"](trade)

    def _handle_order_book(self, order_book):
        """Обработка стакана"""
        if "on_order_book" in self._event_handlers:
            self._event_handlers["on_order_book"](order_book)

    def _handle_last_price(self, last_price):
        """Обработка последней цены"""
        if "on_last_price" in self._event_handlers:
            self._event_handlers["on_last_price"](last_price)

    def _handle_trading_status(self, trading_status):
        """Обработка статуса торгов"""
        if "on_trading_status" in self._event_handlers:
            self._event_handlers["on_trading_status"](trading_status)

    def _handle_portfolio(self, portfolio):
        """Обработка портфеля"""
        if "on_portfolio" in self._event_handlers:
            self._event_handlers["on_portfolio"](portfolio)

    def _handle_position(self, position):
        """Обработка позиции"""
        if "on_position" in self._event_handlers:
            self._event_handlers["on_position"](position)

    def _handle_order_trades(self, order_trades):
        """Обработка сделок по заявкам"""
        if "on_order_trades" in self._event_handlers:
            self._event_handlers["on_order_trades"](order_trades)

    def subscribe_candles(self, class_code: str, symbol: str, timeframe: str):
        """
        Подписываемся на свечи.

        Parameters
        ----------
        class_code : str
            Код режима торгов.
        symbol : str
            Тикер инструмента.
        timeframe : str
            Временной интервал (M1, M5, H1, D1 и т.д.).
        """
        self._provider.subscription_marketdata_queue.put(
            self._provider.create_candle_subscription(class_code, symbol, timeframe)
        )

    def subscribe_order_book(self, class_code: str, symbol: str, depth: int = 10):
        """
        Подписываемся на стакан.

        Parameters
        ----------
        class_code : str
            Код режима торгов.
        symbol : str
            Тикер инструмента.
        depth : int
            Глубина стакана.
        """
        self._provider.subscription_marketdata_queue.put(
            self._provider.create_order_book_subscription(class_code, symbol, depth)
        )

    def subscribe_trades(self, class_code: str, symbol: str):
        """
        Подписываемся на сделки.

        Parameters
        ----------
        class_code : str
            Код режима торгов.
        symbol : str
            Тикер инструмента.
        """
        self._provider.subscription_marketdata_queue.put(
            self._provider.create_last_price_subscription(class_code, symbol)
        )
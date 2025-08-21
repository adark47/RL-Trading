# strategy.py
import torch
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta
from nautilus_trader.common.enums import LogColor
from decimal import Decimal
import pandas as pd
#import fireducks.pandas as pd
#from loguru import logger
from tabulate import tabulate
from datetime import datetime, timedelta  # Дата и время, временной интервал
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators.average.ma_factory import MovingAverageFactory
from nautilus_trader.indicators.average.moving_average import MovingAverageType
from nautilus_trader.indicators.average.hma import HullMovingAverage
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.book import BookLevel
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.functions import order_side_to_str
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import OrderList
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.cache.config import CacheConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.core.data import Data
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.orders import LimitOrder


import talib


class StrategyConfig(StrategyConfig, frozen=True):
    instrument: Instrument
    primary_bar_type: BarType
    trade_size: Decimal



class Strategy(Strategy):

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Инициализация данных
        self.dfBars = pd.DataFrame()
        self.dfBars = pd.DataFrame(columns=['time', 'open', 'close', 'high', 'low', 'volume'])
        self.dfBars.set_index('time', inplace=True)
        self.dfBars.index = pd.to_datetime(self.dfBars.index)
        self.dfBars.index.name = 'time'

        #self.dfBars = pd.DataFrame(columns=['open', 'close', 'high', 'low', 'volume'])
        #self.dfBars.index.name = 'time'
        self.current_position = None
        self.entry_price = None
        self.last_action = 0  # 0=hold по умолчанию







    def on_start(self):
        # Connect indicators with bar-type for automatic updating
        #print(self.config.instrument)
        instrument_id = self.config.instrument

        # Subscribe to bars
        self.subscribe_bars(self.config.primary_bar_type)


        from nautilus_trader.model.enums import BookType

#        self.subscribe_order_book_at_interval(
#                 instrument_id=instrument_id,
#                 book_type=BookType.L2_MBP, # L3_MBO # L2_MBP # L1_MBP
#                 depth=200,
#                 interval_ms=500)

#        self.subscribe_order_book_deltas(instrument_id=instrument_id)
#        self.subscribe_quote_ticks(instrument_id=instrument_id)
#        self.subscribe_trade_ticks(instrument_id=instrument_id)


        self.request_bars(BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"))
#        self.request_quote_ticks(instrument_id=instrument_id)
#        self.request_trade_ticks(instrument_id=instrument_id)
#        #self.request_order_book_snapshot(instrument_id=instrument_id)

    def on_bar(self, bar: Bar):
        self.log.info(f"Bar: {repr(bar)}", color=LogColor.GREEN)

        self.dfBars.loc[pd.to_datetime(pd.to_datetime(datetime.fromtimestamp(bar.ts_init / 1e9).strftime('%Y-%m-%dT%H:%M')))] = [
                                                                             bar.open,
                                                                             bar.close,
                                                                             bar.high,
                                                                             bar.low,
                                                                             bar.volume
                                                                             ]
        self.dfBars = self.dfBars[(datetime.today() - timedelta(days=2)):]  # делаем глубину датафрейма

        # Сохраняем исходные индексы для последующего восстановления
        original_index = df.index

        # Сбрасываем индекс для корректной работы TA-Lib
        self.dfBars = self.dfBars.reset_index(drop=True)

        # Рассчитываем MACD (12,26,9)
        self.logger.debug("Рассчитываем MACD (12,26,9) 📊")
        macd, macd_signal, macd_hist = talib.MACD(self.dfBars['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.dfBars['macd'] = macd
        self.dfBars['macd_signal'] = macd_signal
        self.dfBars['macd_hist'] = macd_hist

        # Рассчитываем RSI (14)
        self.logger.debug("Рассчитываем RSI (14) 📊")
        self.dfBars['rsi'] = talib.RSI(df['close'], timeperiod=14)

        # Рассчитываем Bollinger Bands (20, 2)
        self.logger.debug("Рассчитываем Bollinger Bands (20) 📊")
        upper, middle, lower = talib.BBANDS(self.dfBars['close'], timeperiod=20)
        self.dfBars['bb_upper'] = upper
        self.dfBars['bb_mid'] = middle
        self.dfBars['bb_lower'] = lower

        # Рассчитываем две HMA с разными периодами (9 и 21)
        self.logger.debug("Рассчитываем Hull Moving Average (HMA) с периодами 9 и 21 📐")
        self.dfBars['hma_fast'] = self.calculate_hma(self.dfBars['close'], 9)
        self.dfBars['hma_slow'] = self.calculate_hma(self.dfBars['close'], 21)

        # Рассчитываем ATR (14)
        self.logger.debug("Рассчитываем Average True Range (ATR) с периодом 14 📐")
        atr = talib.ATR(self.dfBars['high'], self.dfBars['low'], self.dfBars['close'], 14)
        self.dfBars['atr'] = atr

        # Восстанавливаем исходные индексы
        self.dfBars.index = original_index


        self.log.info(f"\n {tabulate(self.dfBars.tail(100), showindex=True, headers='keys', tablefmt='psql')}", color=LogColor.GREEN)
        self.log.info(f'Bar info: \n {self.dfBars.info()}', LogColor.RED)
        self.log.info(f'Bar Size: {self.dfBars.memory_usage().sum() / 1024 / 1024:.3f} MB', LogColor.RED)


        # Wait until all registered indicators are initialized
        if not self.indicators_initialized():
            count_of_bars = self.cache.bar_count(self.config.primary_bar_type)
            self.log.info(
                f"Waiting for indicators to warm initialize. | Bars count {count_of_bars}",
                color=LogColor.BLUE,
            )
            return



    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        # Optionally implement


    def on_reset(self) -> None:
        """
        Actions to be performed when the strategy is reset.
        """
        # Optionally implement

    def on_dispose(self) -> None:
        """
        Actions to be performed when the strategy is disposed.

        Cleanup any resources used by the strategy here.

        """
        # Optionally implement

    def on_save(self) -> dict[str, bytes]:
        """
        Actions to be performed when the strategy is saved.

        Create and return a state dictionary of values to be saved.

        Returns
        -------
        dict[str, bytes]
            The strategy state dictionary.

        """
        return {}  # Optionally implement

    def on_load(self, state: dict[str, bytes]) -> None:
        """
        Actions to be performed when the strategy is loaded.

        Saved state values will be contained in the give state dictionary.

        Parameters
        ----------
        state : dict[str, bytes]
            The strategy state dictionary.

        """
        # Optionally implement

    def on_instrument(self, instrument: Instrument) -> None:
        """
        Actions to be performed when the strategy is running and receives an instrument.

        Parameters
        ----------
        instrument : Instrument
            The instrument received.

        """
        # Optionally implement

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """
        Actions to be performed when the strategy is running and receives a quote tick.

        Parameters
        ----------
        tick : QuoteTick
            The tick received.

        """
        # Optionally implement

    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        Actions to be performed when the strategy is running and receives a trade tick.

        Parameters
        ----------
        tick : TradeTick
            The tick received.

        """
        # Optionally implement

    def buy(self) -> None:
        """
        Users simple buy method (example).
        """
        # Optionally implement

    def sell(self) -> None:
        """
        Users simple sell method (example).
        """
        # Optionally implement

    def on_data(self, data: Data) -> None:
        """
        Actions to be performed when the strategy is running and receives data.

        Parameters
        ----------
        data : Data
            The data received.

        """
        # Optionally implement

    def on_event(self, event: Event) -> None:
        """
        Actions to be performed when the strategy is running and receives an event.

        Parameters
        ----------
        event : Event
            The event received.

        """
        # Optionally implement

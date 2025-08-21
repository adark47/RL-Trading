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
from nautilus_trader.model.enums import OrderSide, OrderType, OrderStatus
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
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import LimitOrder

import talib


class StrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId  # Изменено с Instrument на InstrumentId
    primary_bar_type: BarType
    trade_size: Decimal
    warmup_bars: int = 35  # Увеличено до 35 для правильного расчета MACD(12,26,9)


class Strategy(Strategy):

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Инициализация данных
        self.dfBars = pd.DataFrame(columns=['open', 'close', 'high', 'low', 'volume'])
        self.dfBars.index.name = 'time'
        self.current_position = None
        self.entry_price = None
        self.last_action = 0  # 0=hold по умолчанию
        self.warmup_complete = False

    def on_start(self):
        # Подписываемся на бары
        self.subscribe_bars(self.config.primary_bar_type)
        self.log.info(f"Strategy started and subscribed to {self.config.primary_bar_type}", color=LogColor.BLUE)

    def on_bar(self, bar: Bar):
        self.log.info(f"Bar: {repr(bar)}", color=LogColor.GREEN)

        # Преобразуем временную метку бара
        bar_time = pd.Timestamp(bar.ts_init)

        # Добавляем новый бар в DataFrame
        new_row = pd.DataFrame([{
            'open': float(bar.open),
            'close': float(bar.close),
            'high': float(bar.high),
            'low': float(bar.low),
            'volume': float(bar.volume)
        }], index=[bar_time])

        # Исправление предупреждения FutureWarning: добавляем строку без конкатенации с пустым DataFrame
        if self.dfBars.empty:
            self.dfBars = new_row
        else:
            self.dfBars = pd.concat([self.dfBars, new_row])

        # Ограничиваем размер DataFrame (сохраняем последние N баров)
        if len(self.dfBars) > self.config.warmup_bars + 10:
            self.dfBars = self.dfBars.tail(self.config.warmup_bars + 10)

        # Проверяем, достаточно ли данных для расчета индикаторов
        if len(self.dfBars) < self.config.warmup_bars:
            self.log.info(f"Warming up... {len(self.dfBars)}/{self.config.warmup_bars} bars collected",
                          color=LogColor.BLUE)
            return

        if not self.warmup_complete:
            self.log.info(f"Warmup complete! {len(self.dfBars)} bars collected",
                          color=LogColor.GREEN)
            self.warmup_complete = True

        # Рассчитываем индикаторы
        try:
            # Рассчитываем MACD (12,26,9)
            macd, macd_signal, macd_hist = talib.MACD(self.dfBars['close'],
                                                      fastperiod=12,
                                                      slowperiod=26,
                                                      signalperiod=9)
            self.dfBars['macd'] = macd
            self.dfBars['macd_signal'] = macd_signal
            self.dfBars['macd_hist'] = macd_hist

            # Рассчитываем RSI (14)
            self.dfBars['rsi'] = talib.RSI(self.dfBars['close'], timeperiod=14)

            # Рассчитываем Bollinger Bands (20, 2)
            upper, middle, lower = talib.BBANDS(self.dfBars['close'], timeperiod=20)
            self.dfBars['bb_upper'] = upper
            self.dfBars['bb_mid'] = middle
            self.dfBars['bb_lower'] = lower

            # Рассчитываем ATR (14)
            self.dfBars['atr'] = talib.ATR(self.dfBars['high'],
                                           self.dfBars['low'],
                                           self.dfBars['close'],
                                           14)

            # Проверяем наличие NaN в последних значениях индикаторов
            latest = self.dfBars.iloc[-1]
            if latest.isna().any():
                self.log.info("Waiting for complete indicator values...", color=LogColor.BLUE)
                return

            # Проверяем сигналы и размещаем ордера
            self.check_signals(bar)

            # Логируем последние 5 баров для отладки
            if len(self.dfBars) > 5:
                self.log.info(f"\n{tabulate(self.dfBars.tail(5), showindex=True, headers='keys', tablefmt='psql')}",
                              color=LogColor.CYAN)
                self.log.info(f'Memory usage: {self.dfBars.memory_usage().sum() / 1024 / 1024:.3f} MB',
                              color=LogColor.YELLOW)

        except Exception as e:
            self.log.error(f"Error calculating indicators: {str(e)}", color=LogColor.RED)
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)

    def check_signals(self, bar: Bar):
        """Проверяет сигналы и размещает ордера при необходимости"""
        # Получаем последние значения индикаторов
        latest = self.dfBars.iloc[-1]
        previous = self.dfBars.iloc[-2]

        # Проверяем позицию
        position = self.parity_position()

        # Сигнал на покупку: MACD пересекает сигнальную линию снизу вверх
        if latest['macd'] > latest['macd_signal'] and previous['macd'] <= previous['macd_signal']:
            self.log.info(" bullish MACD crossover detected!", color=LogColor.GREEN)
            if position <= 0:  # Нет длинной позиции
                self.execute_buy(bar)

        # Сигнал на продажу: MACD пересекает сигнальную линию сверху вниз
        elif latest['macd'] < latest['macd_signal'] and previous['macd'] >= previous['macd_signal']:
            self.log.info(" bearish MACD crossover detected!", color=LogColor.RED)
            if position >= 0:  # Нет короткой позиции
                self.execute_sell(bar)

    def execute_buy(self, bar: Bar):
        """Размещает рыночный ордер на покупку"""
        # Преобразуем Decimal в Quantity
        quantity = Quantity.from_str(str(self.config.trade_size))
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,  # Используем instrument_id напрямую
            order_side=OrderSide.BUY,
            quantity=quantity,
        )
        self.submit_order(order)
        self.log.info(f"Submitted BUY order: {order}", color=LogColor.GREEN)

    def execute_sell(self, bar: Bar):
        """Размещает рыночный ордер на продажу"""
        # Преобразуем Decimal в Quantity
        quantity = Quantity.from_str(str(self.config.trade_size))
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,  # Используем instrument_id напрямую
            order_side=OrderSide.SELL,
            quantity=quantity,
        )
        self.submit_order(order)
        self.log.info(f"Submitted SELL order: {order}", color=LogColor.RED)

    def parity_position(self) -> int:
        """Возвращает знак чистой позиции: -1 (короткая), 0 (нет позиции), 1 (длинная)"""
        # Используем instrument_id напрямую, без .id
        position = self.portfolio.net_position(self.config.instrument_id)
        if position > 0:
            return 1
        elif position < 0:
            return -1
        return 0

    def on_stop(self) -> None:
        """Действия при остановке стратегии"""
        self.log.info("Strategy stopped", color=LogColor.BLUE)

    def on_reset(self) -> None:
        """Действия при сбросе стратегии"""
        self.dfBars = pd.DataFrame(columns=['open', 'close', 'high', 'low', 'volume'])
        self.dfBars.index.name = 'time'
        self.current_position = None
        self.entry_price = None
        self.last_action = 0
        self.warmup_complete = False
        self.log.info("Strategy reset", color=LogColor.BLUE)

    def on_dispose(self) -> None:
        """Очистка ресурсов"""
        self.log.info("Strategy disposed", color=LogColor.BLUE)
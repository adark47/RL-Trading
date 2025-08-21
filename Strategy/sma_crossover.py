from decimal import Decimal
from typing import List

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import TradingStrategy


class SMACrossoverStrategy(TradingStrategy):
    """
    Стратегия на основе пересечения двух скользящих средних (SMA)

    Покупает, когда быстрая SMA пересекает медленную SMA снизу вверх
    Продает, когда быстрая SMA пересекает медленную SMA сверху вниз
    """

    def __init__(
            self,
            instrument_id: str,
            bar_type: str,
            fast_period: int = 10,
            slow_period: int = 30,
            trade_size: float = 10000.0,
            **kwargs
    ):
        """
        Инициализация стратегии.

        Args:
            instrument_id: Идентификатор инструмента (например, "XYZ/USD")
            bar_type: Тип баров (например, "XYZ/USD.SIM-1-MINUTE-LAST-INTERNAL")
            fast_period: Период быстрой SMA
            slow_period: Период медленной SMA
            trade_size: Размер каждой сделки
        """
        super().__init__(**kwargs)
        self.instrument_id = InstrumentId.from_str(instrument_id)
        self.bar_type = BarType.from_str(bar_type)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trade_size = trade_size

        # История цен для расчета SMA
        self.prices = []

    def on_start(self):
        """Подписка на данные после старта стратегии"""
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Стратегия SMA Crossover запущена для {self.instrument_id.symbol}")

    def on_bar(self, bar: Bar):
        """Обработка новых баров"""
        # Добавляем цену закрытия в историю
        self.prices.append(float(bar.close))

        # Ограничиваем размер истории
        if len(self.prices) > self.slow_period:
            self.prices.pop(0)

        # Ждем накопления достаточного количества данных
        if len(self.prices) < self.slow_period:
            return

        # Рассчитываем текущие SMA
        fast_sma = sum(self.prices[-self.fast_period:]) / self.fast_period
        slow_sma = sum(self.prices[-self.slow_period:]) / self.slow_period

        # Проверяем, есть ли данные для предыдущих значений SMA
        if len(self.prices) <= max(self.fast_period, self.slow_period):
            return

        # Рассчитываем предыдущие SMA
        prev_fast_sma = sum(self.prices[-self.fast_period - 1:-1]) / self.fast_period
        prev_slow_sma = sum(self.prices[-self.slow_period - 1:-1]) / self.slow_period

        # Проверяем пересечение SMA
        if fast_sma > slow_sma and prev_fast_sma <= prev_slow_sma:
            self.log.info(f"ЗОЛОТОЕ ПЕРЕСЕЧЕНИЕ: {fast_sma:.5f} > {slow_sma:.5f}")
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_str(str(self.trade_size)),
                )
            )

        elif fast_sma < slow_sma and prev_fast_sma >= prev_slow_sma:
            self.log.info(f"СМЕРТНОЕ ПЕРЕСЕЧЕНИЕ: {fast_sma:.5f} < {slow_sma:.5f}")
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=Quantity.from_str(str(self.trade_size)),
                )
            )
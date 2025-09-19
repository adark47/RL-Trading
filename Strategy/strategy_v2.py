# Strategy_v2.py

import os
import sys
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import platform
try:
    # Пытаемся импортировать fireducks.pandas только на Linux
    if platform.system().lower() == 'linux':
        import fireducks.pandas as pd
        print("Загружен fireducks.pandas")
    else:
        raise ImportError
except ImportError:
    import pandas as pd
    print("Загружен стандартный pandas")

import torch
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce, PositionSide
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId, PositionId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.core.rust.model import OrderSide

# --- Импорты для модели ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем TA стратегию (без RL агента)
from data.ta_strategy import apply_strategy

# --- Оптимизация: Импорт collections для deque ---
import collections

# --- Импорт для логирования в файл ---
import logging


# -------------------------------------------------

class StrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    primary_bar_type: BarType
    trade_size: Decimal
    trade_mode: str
    ta_config_path: str = '../data/ta_config_optimized.json'  # Путь к конфигурации TA
    version: str = 'v2_with_TA_only'


class Strategy(Strategy):

    def __init__(self, config: StrategyConfig):  # Уточнение типа
        """
        Initialize the TA-based strategy.

        Parameters
        ----------
        config : RLStrategyConfig
            The strategy configuration.
        """
        super().__init__(config)
        self.version = config.version
        self.instrument_id = config.instrument_id
        self.bar_type = config.primary_bar_type
        self.trade_size = config.trade_size

        # self.model_path больше не используется
        self.ta_config_path = config.ta_config_path

        # --- Оптимизация: Используем deque вместо DataFrame для буфера баров ---
        # Для экономии памяти и повышения эффективности добавления/удаления
        # Определяем максимальный размер буфера (немного больше, чем нужно для TA)
        # Используем фиксированное значение или значение из конфига, если доступно
        buffer_size_for_ta = 20000  # Пример, адаптируйте при необходимости
        self.bars_buffer = collections.deque(maxlen=buffer_size_for_ta)
        # -----------------------------------------------------------------------
        self.last_processed_bar_time = None

        self.current_position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0  # Цена входа для расчета unrealized pnl

        self.order_id_counter = 0

        # Определяем ожидаемые колонки TA, включая сигналы
        self.required_ta_columns = [
            'open', 'high', 'low', 'close', 'volume',
            # Сигналы, на которые мы будем реагировать
            'long_entries', 'long_exits', 'short_entries', 'short_exits'
        ]

        # --- Настройка логирования в файл ---
        self.file_logger = logging.getLogger(f"StrategyFileLogger_{self.id}")  # Используем уникальный ID стратегии
        # Проверяем, добавлен ли уже обработчик, чтобы избежать дубликатов при сбросе
        if not self.file_logger.handlers:
            self.file_logger.setLevel(
                logging.DEBUG)  # Установите уровень по необходимости (DEBUG, INFO, WARNING, ERROR)
            # Создаем форматтер
            formatter = logging.Formatter(
                fmt='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S'
            )
            # Создаем обработчик для записи в файл
            # Убедитесь, что директория 'logs' существует
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file_path = os.path.join(log_dir, f"strategy_{self.version}.log")  # Файл лога будет в logs/strategy_debug_<strategy_id>.log
            file_handler = logging.FileHandler(log_file_path, mode='w')  # 'w' для перезаписи при каждом запуске, 'a' для добавления
            file_handler.setFormatter(formatter)
            # Добавляем обработчик к логгеру
            self.file_logger.addHandler(file_handler)
            # Отключаем распространение логов на родительские логгеры (например, корневой), чтобы избежать дублирования
            self.file_logger.propagate = False
        # ------------------------------------

    def on_start(self):
        """
        Вызывается при запуске стратегии.
        """
        self.log.info("TA Strategy starting...", color=LogColor.MAGENTA)
        self.log.info(
            f"Strategy Config: Instrument={self.instrument_id}, BarType={self.bar_type}, TradeSize={self.trade_size}, TAConfigPath={self.ta_config_path}",
            color=LogColor.BLUE)
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Subscribed to bars: {self.bar_type}", color=LogColor.BLUE)

        self.log.info("TA strategy ready to process bars based on signals.", color=LogColor.GREEN)

    # _initialize_agent больше не нужен

    def on_bar(self, bar: Bar):
        """
        Actions to be performed when the strategy receives a bar.
        Updates data, applies TA strategy, and executes trades based on signals.

        Parameters
        ----------
        bar : Bar
            The received bar.
        """
        # Проверка дубликатов может быть важна
        bar_time = pd.Timestamp(bar.ts_init)  # ts_init обычно более точный

        self.log.debug(
            f"Received bar: Time={bar_time}, Open={bar.open}, High={bar.high}, Low={bar.low}, Close={bar.close}, Volume={bar.volume}",
            color=LogColor.BLUE)
        self.file_logger.debug(
            f"Received bar: Time={bar_time}, Open={bar.open}, High={bar.high}, Low={bar.low}, Close={bar.close}, Volume={bar.volume}")

        if self.last_processed_bar_time is not None and bar_time <= self.last_processed_bar_time:
            self.log.debug(f"Skipping duplicate or old bar: {bar_time} <= {self.last_processed_bar_time}")
            self.file_logger.debug(f"Skipping duplicate or old bar: {bar_time} <= {self.last_processed_bar_time}")
            return

        self.last_processed_bar_time = bar_time
        self.log.debug(f"Processing bar: {bar_time}")
        self.file_logger.debug(f"Processing bar: {bar_time}")

        # --- Оптимизация: Добавление бара в deque ---
        new_bar_dict = {
            'date': bar_time,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume)
        }
        self.bars_buffer.append(new_bar_dict)  # deque автоматически управляет maxlen
        self.log.debug(f"Bar added to buffer. Buffer size: {len(self.bars_buffer)}/{self.bars_buffer.maxlen}")
        self.file_logger.debug(f"Bar added to buffer. Buffer size: {len(self.bars_buffer)}/{self.bars_buffer.maxlen}")
        # ------------------------------------------

        # --- Оптимизация: Проверка достаточности данных ---
        # Используем длину буфера вместо DataFrame
        # Используем фиксированное значение или значение из конфига, если доступно
        min_required_bars = 30  # Пример, адаптируйте при необходимости
        if len(self.bars_buffer) < min_required_bars:
            needed = min_required_bars - len(self.bars_buffer)
            self.log.info(
                f"Not enough data yet. Need {min_required_bars} bars, have {len(self.bars_buffer)}. Waiting for {needed} more bars.")
            self.file_logger.info(
                f"Not enough data yet. Need {min_required_bars} bars, have {len(self.bars_buffer)}. Waiting for {needed} more bars.")
            return
        # -----------------------------------------------

        try:
            # --- Оптимизация: Создание DataFrame из deque только при необходимости ---
            # Создаем DataFrame из буфера перед применением TA стратегии
            df_bars_for_ta = pd.DataFrame(list(self.bars_buffer))  # list(deque) для создания DataFrame
            self.log.debug(f"DataFrame created for TA analysis. Shape: {df_bars_for_ta.shape}")
            self.file_logger.debug(f"DataFrame created for TA analysis. Shape: {df_bars_for_ta.shape}")

            # --- Добавлено для диагностики: логирование последних строк буфера ---
            self.log.debug(f"Last 3 rows of df_bars_for_ta:\n{df_bars_for_ta.tail(3)}")
            self.file_logger.debug(f"Last 3 rows of df_bars_for_ta:\n{df_bars_for_ta.tail(3).to_string()}")
            # ---------------------------------------------------------------------

            # Применяем стратегию технического анализа
            self.log.debug("Applying TA strategy...")
            self.file_logger.debug("Applying TA strategy...")
            df_with_ta = apply_strategy(df_bars_for_ta, config_path=self.ta_config_path)
            self.log.debug(f"TA strategy applied. Resulting DataFrame shape: {df_with_ta.shape}")
            self.file_logger.debug(f"TA strategy applied. Resulting DataFrame shape: {df_with_ta.shape}")

            # --- Добавлено для диагностики: логирование формы и последних строк результата ---
            self.log.debug(f"df_with_ta columns: {df_with_ta.columns.tolist()}")
            self.file_logger.debug(f"df_with_ta columns: {df_with_ta.columns.tolist()}")
            self.log.debug(f"Last 3 rows of df_with_ta (with signals):\n{df_with_ta.tail(3)}")
            self.file_logger.debug(f"Last 3 rows of df_with_ta (with signals):\n{df_with_ta.tail(3).to_string()}")
            # ---------------------------------------------------------------------------------------

            missing_cols = [col for col in self.required_ta_columns if col not in df_with_ta.columns]
            if missing_cols:
                self.log.error(f"Missing required TA columns: {missing_cols}")
                self.file_logger.error(f"Missing required TA columns: {missing_cols}")
                return  # Прекращаем обработку, если данные неполные
            else:
                self.log.debug("All required TA columns are present.")
                self.file_logger.debug("All required TA columns are present.")

            # Получаем последнюю строку данных с сигналами
            if not df_with_ta.empty:
                last_row = df_with_ta.iloc[-1]
                self.log.debug(f"Last row data: {last_row.to_dict()}")
                self.file_logger.debug(f"Last row data: {last_row.to_dict()}")

                # --- Усиленная проверка и логирование ---
                le_val = last_row['long_entries']
                se_val = last_row['short_entries']
                lx_val = last_row['long_exits']
                sx_val = last_row['short_exits']

                self.log.debug(f"[DEBUG CHECK] BarTime: {bar_time}")
                self.file_logger.debug(f"[DEBUG CHECK] BarTime: {bar_time}")
                self.log.debug(
                    f"[DEBUG CHECK] Signals Raw: LE={le_val} (type={type(le_val)}), SE={se_val} (type={type(se_val)}), LX={lx_val} (type={type(lx_val)}), SX={sx_val} (type={type(sx_val)})")
                self.file_logger.debug(
                    f"[DEBUG CHECK] Signals Raw: LE={le_val} (type={type(le_val)}), SE={se_val} (type={type(se_val)}), LX={lx_val} (type={type(lx_val)}), SX={sx_val} (type={type(sx_val)})")
                self.log.debug(f"[DEBUG CHECK] Current Pos: {self.current_position}")
                self.file_logger.debug(f"[DEBUG CHECK] Current Pos: {self.current_position}")

                # Явная и пошаговая проверка
                is_le_true = le_val == True or (hasattr(le_val, 'item') and le_val.item() == True) or (
                            isinstance(le_val, (int, float, np.integer, np.floating)) and le_val == 1)
                is_se_true = se_val == True or (hasattr(se_val, 'item') and se_val.item() == True) or (
                            isinstance(se_val, (int, float, np.integer, np.floating)) and se_val == 1)
                is_lx_true = lx_val == True or (hasattr(lx_val, 'item') and lx_val.item() == True) or (
                            isinstance(lx_val, (int, float, np.integer, np.floating)) and lx_val == 1)
                is_sx_true = sx_val == True or (hasattr(sx_val, 'item') and sx_val.item() == True) or (
                            isinstance(sx_val, (int, float, np.integer, np.floating)) and sx_val == 1)

                self.log.debug(
                    f"[DEBUG CHECK] Evaluated: is_le_true={is_le_true}, is_se_true={is_se_true}, is_lx_true={is_lx_true}, is_sx_true={is_sx_true}")
                self.file_logger.debug(
                    f"[DEBUG CHECK] Evaluated: is_le_true={is_le_true}, is_se_true={is_se_true}, is_lx_true={is_lx_true}, is_sx_true={is_sx_true}")
                # --- Конец усиленной проверки ---

                # Проверяем сигналы и выполняем торговлю
                signal_processed = False
                self.log.debug(f"Current position before signal check: {self._get_position_string()}")
                self.file_logger.debug(f"Current position before signal check: {self._get_position_string()}")

                # Проверка сигналов закрытия
                if is_lx_true and self.current_position == 1:
                    self.log.info("Signal: Long Exit -> Selling.", color=LogColor.CYAN)
                    self.file_logger.info("Signal: Long Exit -> Selling.")
                    self.sell()
                    signal_processed = True
                elif is_sx_true and self.current_position == -1:
                    self.log.info("Signal: Short Exit -> Buying.", color=LogColor.CYAN)
                    self.file_logger.info("Signal: Short Exit -> Buying.")
                    self.buy()
                    signal_processed = True

                # Проверка сигналов открытия (только если нет открытой позиции)
                elif self.current_position == 0:
                    self.log.debug("Inside 'current_position == 0' block")
                    self.file_logger.debug("Inside 'current_position == 0' block")
                    if is_le_true:  # <--- Используем результат усиленной проверки
                        self.log.info("Signal: Long Entry -> Buying.", color=LogColor.CYAN)
                        self.file_logger.info("Signal: Long Entry -> Buying.")
                        self.buy()
                        signal_processed = True
                    elif is_se_true:  # <--- Используем результат усиленной проверки
                        self.log.info("Signal: Short Entry -> Selling.", color=LogColor.CYAN)
                        self.file_logger.info("Signal: Short Entry -> Selling.")
                        self.sell()
                        signal_processed = True
                    else:
                        self.log.debug(f"No entry signal triggered. is_le_true: {is_le_true}, is_se_true: {is_se_true}")
                        self.file_logger.debug(
                            f"No entry signal triggered. is_le_true: {is_le_true}, is_se_true: {is_se_true}")
                else:
                    self.log.debug(
                        f"Not processing entry signals because current_position is not 0. Current position: {self.current_position}")
                    self.file_logger.debug(
                        f"Not processing entry signals because current_position is not 0. Current position: {self.current_position}")

                if not signal_processed:
                    # Отключаем это сообщение для чистоты лога, так как оно появляется часто
                    # self.log.info("No active signal. No trade executed.", color=LogColor.CYAN)
                    # self.file_logger.info("No active signal. No trade executed.")
                    pass

                # --- Конец логики торговли ---
                self.log.debug(
                    f"Bar processing completed for {bar_time}. Current position: {self._get_position_string()}")
                self.file_logger.debug(
                    f"Bar processing completed for {bar_time}. Current position: {self._get_position_string()}")

            else:
                self.log.warning("TA DataFrame is empty after applying strategy.")
                self.file_logger.warning("TA DataFrame is empty after applying strategy.")
                return

        except Exception as e:
            self.log.error(f"Error in on_bar processing: {str(e)}", color=LogColor.RED)
            self.file_logger.error(f"Error in on_bar processing: {str(e)}",
                                   exc_info=True)  # exc_info=True добавит трассировку стека
            import traceback
            self.log.debug(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)
            self.file_logger.debug(f"Traceback: {traceback.format_exc()}")

    def buy(self) -> None:
        """
        Отправляет рыночный ордер на покупку.
        (Функция не изменена, как требовалось)
        """
        self.log.info("Initiating BUY action...", color=LogColor.GREEN)
        self.file_logger.info("Initiating BUY action...")
        # Создаем уникальный ID для ордера
        self.order_id_counter += 1
        client_order_id = ClientOrderId(f"RL_BUY_{self.order_id_counter}")

        # Создаем рыночный ордер на покупку
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str(str(self.trade_size)),  # Конвертируем Decimal в Quantity
            client_order_id=client_order_id,
            # time_in_force по умолчанию GTC для рыночных ордеров, можно опустить
        )

        # Отправляем ордер
        self.log.info(f"Submitting BUY order: {order}", color=LogColor.GREEN)
        self.file_logger.info(f"Submitting BUY order: {order}")
        self.submit_order(order)
        self.log.info(f"BUY order submitted: {order.client_order_id}", color=LogColor.GREEN)
        self.file_logger.info(f"BUY order submitted: {order.client_order_id}")

    def sell(self) -> None:
        """
        Отправляет рыночный ордер на продажу.
        (Функция не изменена, как требовалось)
        """
        self.log.info("Initiating SELL action...", color=LogColor.YELLOW)
        self.file_logger.info("Initiating SELL action...")
        # Создаем уникальный ID для ордера
        self.order_id_counter += 1
        client_order_id = ClientOrderId(f"RL_SELL_{self.order_id_counter}")

        # Создаем рыночный ордер на продажу
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_str(str(self.trade_size)),  # Конвертируем Decimal в Quantity
            client_order_id=client_order_id,
            # time_in_force по умолчанию GTC для рыночных ордеров, можно опустить
        )

        # Отправляем ордер
        self.log.info(f"Submitting SELL order: {order}", color=LogColor.YELLOW)
        self.file_logger.info(f"Submitting SELL order: {order}")
        self.submit_order(order)
        self.log.info(f"SELL order submitted: {order.client_order_id}",
                      color=LogColor.YELLOW)  # Используем другой цвет для различия
        self.file_logger.info(f"SELL order submitted: {order.client_order_id}")

    def on_order_filled(self, order_filled):
        """
        Обновляет внутреннее состояние при исполнении ордера.
        """
        self.log.info(f"Order filled event received: {order_filled}", color=LogColor.MAGENTA)
        self.file_logger.info(f"Order filled event received: {order_filled}")
        self.log.info(
            f"Filled details - ClientOrderId: {order_filled.client_order_id}, InstrumentId: {order_filled.instrument_id}, Qty: {order_filled.last_qty}, Price: {order_filled.last_px}, Side: {order_filled.order_side}",
            color=LogColor.MAGENTA)
        self.file_logger.info(
            f"Filled details - ClientOrderId: {order_filled.client_order_id}, InstrumentId: {order_filled.instrument_id}, Qty: {order_filled.last_qty}, Price: {order_filled.last_px}, Side: {order_filled.order_side}")

        # Получаем исполненный ордер из кэша
        order = self.cache.order(order_filled.client_order_id)
        if order is None:
            self.log.warning(f"Filled order {order_filled.client_order_id} not found in cache.")
            self.file_logger.warning(f"Filled order {order_filled.client_order_id} not found in cache.")
            return

        # Получаем цену исполнения
        fill_price = order_filled.last_px.as_double()  # Предполагаем, что это цена
        fill_qty = order_filled.last_qty.as_double()
        self.log.info(f"Fill confirmed: Price={fill_price}, Quantity={fill_qty}", color=LogColor.CYAN)
        self.file_logger.info(f"Fill confirmed: Price={fill_price}, Quantity={fill_qty}")

        # Логируем состояние до обновления
        self.log.debug(
            f"State before fill update - Position: {self._get_position_string()}, EntryPrice: {self.entry_price}")
        self.file_logger.debug(
            f"State before fill update - Position: {self._get_position_string()}, EntryPrice: {self.entry_price}")

        # Определяем направление действия на основе стороны ордера
        if order.side == OrderSide.BUY:
            # Покупка может означать открытие лонга или закрытие шорта
            if self.current_position == 0:  # Открываем лонг
                self.current_position = 1
                self.entry_price = fill_price
                self.log.info(f"Opened LONG position at price {self.entry_price}", color=LogColor.GREEN)
                self.file_logger.info(f"Opened LONG position at price {self.entry_price}")
            elif self.current_position == -1:  # Закрываем шорт
                self.current_position = 0
                self.entry_price = 0.0
                realized_pnl = (self.entry_price - fill_price) * fill_qty  # Простой расчет PnL для шорта
                self.log.info(f"Closed SHORT position. Realized PnL: {realized_pnl:.5f}", color=LogColor.GREEN)
                self.file_logger.info(f"Closed SHORT position. Realized PnL: {realized_pnl:.5f}")
            else:
                self.log.warning(f"BUY fill received, but current_position is unexpected: {self.current_position}")
                self.file_logger.warning(
                    f"BUY fill received, but current_position is unexpected: {self.current_position}")

        elif order.side == OrderSide.SELL:
            # Продажа может означать открытие шорта или закрытие лонга
            if self.current_position == 0:  # Открываем шорт
                self.current_position = -1
                self.entry_price = fill_price
                self.log.info(f"Opened SHORT position at price {self.entry_price}", color=LogColor.YELLOW)
                self.file_logger.info(f"Opened SHORT position at price {self.entry_price}")
            elif self.current_position == 1:  # Закрываем лонг
                self.current_position = 0
                self.entry_price = 0.0
                realized_pnl = (fill_price - self.entry_price) * fill_qty  # Простой расчет PnL для лонга
                self.log.info(f"Closed LONG position. Realized PnL: {realized_pnl:.5f}", color=LogColor.YELLOW)
                self.file_logger.info(f"Closed LONG position. Realized PnL: {realized_pnl:.5f}")
            else:
                self.log.warning(f"SELL fill received, but current_position is unexpected: {self.current_position}")
                self.file_logger.warning(
                    f"SELL fill received, but current_position is unexpected: {self.current_position}")
        else:
            self.log.warning(f"Unknown order side filled: {order.side}")
            self.file_logger.warning(f"Unknown order side filled: {order.side}")

        # Логируем состояние после обновления
        self.log.debug(
            f"State after fill update - Position: {self._get_position_string()}, EntryPrice: {self.entry_price}")
        self.file_logger.debug(
            f"State after fill update - Position: {self._get_position_string()}, EntryPrice: {self.entry_price}")

    def on_stop(self):
        """
        Вызывается при остановке стратегии.
        """
        self.log.info(
            f"TA Strategy stopped. Final state - Position: {self._get_position_string()}, EntryPrice: {self.entry_price}",
            color=LogColor.MAGENTA)
        self.file_logger.info(
            f"TA Strategy stopped. Final state - Position: {self._get_position_string()}, EntryPrice: {self.entry_price}")

    def on_reset(self):
        """
        Сбрасывает внутреннее состояние стратегии.
        """
        self.log.info(
            f"TA Strategy reset initiated. Current state - Position: {self._get_position_string()}, BufferSize: {len(self.bars_buffer)}",
            color=LogColor.MAGENTA)
        self.file_logger.info(
            f"TA Strategy reset initiated. Current state - Position: {self._get_position_string()}, BufferSize: {len(self.bars_buffer)}")
        # --- Оптимизация: Очистка deque ---
        self.bars_buffer.clear()
        # ---------------------------------
        self.last_processed_bar_time = None
        self.current_position = 0  # Убедиться, что позиция сбрасывается
        self.entry_price = 0.0  # Сброс цены входа
        # Инициализируем историю действий заново (если используется в других местах)
        # self.history_actions = []
        self.order_id_counter = 0
        self.log.info("TA Strategy reset completed.", color=LogColor.MAGENTA)
        self.file_logger.info("TA Strategy reset completed.")

    def on_dispose(self):
        """
        Вызывается при уничтожении стратегии.
        """
        self.log.info(
            f"TA Strategy disposed. Final state - Position: {self._get_position_string()}, BufferSize: {len(self.bars_buffer)}",
            color=LogColor.MAGENTA)
        self.file_logger.info(
            f"TA Strategy disposed. Final state - Position: {self._get_position_string()}, BufferSize: {len(self.bars_buffer)}")
        # --- Оптимизация: Очистка deque при уничтожении ---
        self.bars_buffer.clear()
        # -------------------------------------------------

    def _get_position_string(self):
        """Вспомогательный метод для получения строкового представления позиции."""
        if self.current_position == 1:
            return "LONG"
        elif self.current_position == -1:
            return "SHORT"
        else:
            return "FLAT"

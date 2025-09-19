# strategy_v3.py
# Импортируем необходимые библиотеки
import os
import sys
from decimal import Decimal  # Для точных денежных расчетов
from datetime import datetime, timedelta
import numpy as np  # Для быстрых числовых операций
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
  # Для удобной работы с временными рядами
import collections
import logging  # Для ведения логов в файл
from tabulate import tabulate  # Для красивого форматирования таблиц в логах
import time  # Для измерения времени выполнения (профилирования)
# --- Импорты Nautilus Trader ---
# Импортируем базовые классы и перечисления фреймворка
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import  QuoteTick, OrderBookDelta
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce, PositionSide
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import LimitOrder, MarketOrder
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId, PositionId, AccountId
from nautilus_trader.model.objects import Price, Quantity, Money, Currency
from nautilus_trader.trading.strategy import Strategy
# --- Оптимизация: Добавляем путь для импорта TA-логики (если понадобится) ---
# Позволяет импортировать модули из родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- Импорт для логирования в файл ---
import logging
# --- Константы в ПРОЦЕНТАХ ---
# Эти значения используются по умолчанию в конфигурации, если не указаны другие
MAX_BUFFER_SIZE = 10000                     # Максимальное количество тиков для хранения в буфере
MIN_VOLUME_THRESHOLD = 100                  # Минимальный объем на уровне для участия в торговле (в единицах базовой валюты)
MIN_SPREAD_PCT = 0.0005                     # Минимальный спред в процентах от цены (0.0005% = 0.5 bps)
TRAILING_STOP_PCT = 0.0003                  # Трейлинг-стоп в % от текущей цены (0.03%)
TAKE_PROFIT_PCT = 0.0005                    # Тейк-профит в % от цены входа (0.05%)
LOOKBACK_5M = 5 * 60                        # 5 минут в секундах
LOOKBACK_15M = 15 * 60                      # 15 минут в секундах
ORDER_BOOK_DEPTH = 50                       # Глубина стакана для анализа (лучшие 50 уровней)
COMMISSION_PCT = 0.0004                     # Комиссия 0.04% за сделку (0.0004)
# Минимальная прибыль до комиссий, чтобы сделка была потенциально прибыльной
MIN_PROFIT_BEFORE_COMMISSION = COMMISSION_PCT * 2.1  # 0.084% — с запасом 10%
# Максимальное расстояние от текущей цены до уровня поддержки/сопротивления для входа
MAX_ENTRY_DISTANCE_FACTOR = 0.0005          # 0.05% от цены
TRADE_COOLDOWN_SECONDS = 5                  # Минимальный интервал между попытками входа (сек)
MAX_ORDER_AGE_SECONDS = 15                  # Время жизни лимитного ордера до отмены (сек)
MAX_DAILY_LOSS_PCT = 0.01                   # Максимальная просадка в % капитала (1%)
POST_TRADE_COOLDOWN_SECONDS = 10            # Коулдаун после закрытия позиции (например, чтобы дать рынку "остыть")
class StrategyConfig(StrategyConfig, frozen=True):
    """
    Конфигурация стратегии скальпинга с лимитными ордерами по стакану.
    Все параметры теперь в процентах — универсально для любых активов.
    """
    instrument_id: InstrumentId  # Идентификатор торгуемого инструмента
    trade_size: Decimal           # Размер лота для одной сделки (например, "0.1")
    primary_bar_type: BarType
    trade_mode: str
    version: str = 'v3_scalping'
    # Периоды для агрегации объемов (в секундах)
    lookback_5m: int = LOOKBACK_5M
    lookback_15m: int = LOOKBACK_15M
    min_volume_threshold: float = MIN_VOLUME_THRESHOLD                  # Минимальный объем на уровне для участия в торговле
    min_spread_pct: float = MIN_SPREAD_PCT                              # Минимальный спред в процентах от цены
    trailing_stop_pct: float = TRAILING_STOP_PCT                        # Трейлинг-стоп в процентах от текущей цены
    take_profit_pct: float = TAKE_PROFIT_PCT                            # Тейк-профит в процентах от цены входа
    order_book_depth: int = ORDER_BOOK_DEPTH                            # Глубина стакана для анализа
    # --- Новые параметры ---
    commission_pct: float = COMMISSION_PCT                              # Комиссия за сделку в процентах
    min_profit_before_commission: float = MIN_PROFIT_BEFORE_COMMISSION  # Минимальная прибыль до комиссий
    max_entry_distance_factor: float = MAX_ENTRY_DISTANCE_FACTOR        # Максимальное расстояние до уровня для входа в процентах от цены
    trade_cooldown_seconds: int = TRADE_COOLDOWN_SECONDS                # Минимальный интервал между входами (сек)
    max_order_age_seconds: int = MAX_ORDER_AGE_SECONDS                  # Время жизни ордера до отмены (сек)
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT                      # Максимальная просадка в процентах от начального капитала
    initial_capital: Decimal = Decimal("10000")                         # 🔴 КРИТИЧЕСКОЕ: Начальный капитал для расчета просадки
    post_trade_cooldown_seconds: int = POST_TRADE_COOLDOWN_SECONDS      # Коулдаун после закрытия позиции (сек)
class Strategy(Strategy):
    """
    Стратегия скальпинга на лимитных ордерах по стакану.
    - Анализирует глубину стакана и объемы за 5 и 15 минут.
    - Выставляет лимитные ордеры на уровнях с высокой ликвидностью.
    - Использует трейлинг-стоп и тейк-профит в процентах от цены.
    - Учитывает спред, объемы, комиссию и время жизни ордера.
    - Визуализация превью через tabulate.
    """
    def __init__(self, config: StrategyConfig):
        """
        Конструктор стратегии. Инициализирует все параметры и внутренние переменные.
        """
        super().__init__(config)
        # --- Настройка логирования в файл ---
        self.file_logger = logging.getLogger(f"StrategyFileLogger_{self.id}")  # Используем уникальный ID стратегии
        # Проверяем, добавлен ли уже обработчик, чтобы избежать дубликатов при сбросе
        if not self.file_logger.handlers:
            self.file_logger.setLevel(
                logging.INFO)  # Установите уровень по необходимости (DEBUG, INFO, WARNING, ERROR)
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
            log_file_path = os.path.join(log_dir, f"strategy_{config.version}.log")     # Файл лога будет в logs/strategy_debug_<strategy_id>.log
            file_handler = logging.FileHandler(log_file_path, mode='w')                 # 'w' для перезаписи при каждом запуске, 'a' для добавления
            file_handler.setFormatter(formatter)
            # Добавляем обработчик к логгеру
            self.file_logger.addHandler(file_handler)
            # Отключаем распространение логов на родительские логгеры (например, корневой), чтобы избежать дублирования
            self.file_logger.propagate = False
        # --- Конфигурация ---
        # Сохраняем параметры из конфига в атрибуты объекта
        self.version = config.version
        self.instrument_id = config.instrument_id
        self.trade_size = config.trade_size
        self.lookback_5m = config.lookback_5m
        self.lookback_15m = config.lookback_15m
        self.min_volume_threshold = config.min_volume_threshold
        self.min_spread_pct = config.min_spread_pct
        self.trailing_stop_pct = config.trailing_stop_pct
        self.take_profit_pct = config.take_profit_pct
        self.order_book_depth = config.order_book_depth
        # --- Новые параметры ---
        self.commission_pct = config.commission_pct
        self.min_profit_before_commission = config.min_profit_before_commission
        self.max_entry_distance_factor = config.max_entry_distance_factor
        self.trade_cooldown_seconds = config.trade_cooldown_seconds
        self.max_order_age_seconds = config.max_order_age_seconds
        self.max_daily_loss_pct = config.max_daily_loss_pct
        self.initial_capital = config.initial_capital
        self.post_trade_cooldown_seconds = config.post_trade_cooldown_seconds
        # --- Состояние стратегии ---
        # 0=нет позиции, 1=лонг, -1=шорт
        self.current_position = 0
        # Цена входа в позицию
        self.entry_price = 0.0
        # Активный уровень трейлинг-стопа
        self.trailing_stop_price = 0.0
        # Уровень тейк-профита
        self.take_profit_price = 0.0
        # Время последнего полученного тика
        self.last_tick_time = None
        # Словарь для отслеживания выставленных, но неисполненных ордеров
        self.pending_orders = {}  # dict[ClientOrderId] = timestamp
        # Счетчик для генерации уникальных ID ордеров
        self.order_id_counter = 0
        # Общий профит и лосс (PnL) в базовой валюте
        self.total_pnl = Decimal("0")
        # Время последнего входа в сделку (для коулдауна)
        self.last_trade_time = None
        # Время последнего закрытия позиции (для коулдауна)
        self.last_close_time = None
        # Текущий размер позиции (для учета частичных исполнений)
        self.position_size = Decimal("0")
        # --- Буферы данных (оптимизация: numpy массив вместо deque) ---
        # Используем numpy массив для быстрого хранения и обработки тиков
        self.ticks_buffer = np.zeros(MAX_BUFFER_SIZE, dtype=[
            ('timestamp', 'datetime64[ns]'),
            ('bid', 'f8'), ('ask', 'f8'),
            ('bid_qty', 'f8'), ('ask_qty', 'f8'),
            ('mid', 'f8'), ('spread_pct', 'f8')
        ])
        # Индекс для записи следующего тика
        self.buffer_idx = 0
        # Флаг, показывающий, что буфер полностью заполнен хотя бы один раз
        self.buffer_filled = False
        # --- Агрегированные данные ---
        # Хранят результаты анализа объемов за 5 и 15 минут
        self.aggregated_5m = None
        self.aggregated_15m = None
        # --- Фиксированные логарифмические бины (оптимизация производительности) ---
        # Предварительно рассчитанные уровни цен для агрегации объемов
        self.fixed_bins = None
        # --- Профилирование ---
        # Словарь для сбора статистики по времени выполнения и количеству операций
        self.profiling = {
            'aggregate_time': 0,        # Время на агрегацию объемов
            'analyze_time': 0,          # Время на анализ стакана и принятие решений
            'update_tp_time': 0,        # Время на обновление стопов/тейков
            'total_ticks': 0,           # Общее количество обработанных тиков
            'submitted_orders': 0,      # Количество отправленных ордеров
            'filled_orders': 0,         # Количество исполненных ордеров
            'canceled_orders': 0        # Количество отмененных ордеров
        }
        # --- Инструмент (будет заполнен при on_start) ---
        self.instrument = None
        self.log.info("ScalpStrategy initialized (Proportional Mode with Commission).", color=LogColor.BLUE)
        self.file_logger.info("ScalpStrategy initialized (Proportional Mode with Commission).")
        # ------------------------------------
    def on_start(self):
        """
        Вызывается при запуске стратегии.
        Подписываемся на необходимые данные и инициализируем инструмент.
        """
        self.log.info("ScalpStrategy starting...", color=LogColor.MAGENTA)
        self.file_logger.info("ScalpStrategy starting...")
        # Подписываемся на котировки (bid/ask) и изменения стакана
        self.subscribe_quote_ticks(self.instrument_id)
        # Исправление: Используем правильную глубину стакана для Bybit Linear (50 вместо 5)
        # self.subscribe_order_book_deltas(self.instrument_id, depth=self.order_book_depth)
        self.subscribe_order_book_deltas(self.instrument_id, depth=50) # Исправлено
        self.log.info(f"Subscribed to quote ticks and order book deltas for {self.instrument_id}", color=LogColor.GREEN)
        self.file_logger.info(f"Subscribed to quote ticks and order book deltas for {self.instrument_id}")
        # Получаем объект инструмента из кэша Nautilus
        self.instrument = self.cache.instrument(self.instrument_id)
        if not self.instrument:
            self.log.error(f"Instrument {self.instrument_id} not found in cache!", color=LogColor.RED)
            self.file_logger.error(f"Instrument {self.instrument_id} not found in cache!")
            return
        self.log.info(f"Instrument: {self.instrument.symbol}, Price increment: {self.instrument.price_increment}", color=LogColor.CYAN)
        self.file_logger.info(f"Instrument: {self.instrument.symbol}, Price increment: {self.instrument.price_increment}")
        # Инициализируем уровни стопов и тейков
        self.trailing_stop_price = 0.0
        self.take_profit_price = 0.0
        self.log.info(f"Initial capital set to: {self.initial_capital:.2f}", color=LogColor.CYAN)
        self.file_logger.info(f"Initial capital set to: {self.initial_capital:.2f}")
        # --- 🔥 КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: Предварительный расчет фиксированных логарифмических бинов ---
        # Это ускоряет агрегацию объемов, так как не нужно пересчитывать бины каждый раз
        # Оцениваем диапазон цен на основе tick_size и средней цены
        # Используем ±20% от средней цены как пределы для бинов
        # --- 🔥 ИСПРАВЛЕНИЕ: Универсальный способ получения оценки цены ---
        # Вместо использования несуществующего price_filter, используем price_increment и оценку цены
        # Например, можно взять минимальную цену как price_increment или немного больше него
        # Или можно попробовать получить последнюю цену из кэша, если она есть
        try:
            # Попробуем получить последнюю цену из кэша (может быть None в начале)
            last_quote = self.cache.quote_tick(self.instrument_id)
            if last_quote:
                mid_estimate = float((last_quote.bid_price + last_quote.ask_price) / 2)
            else:
                # Если котировки еще нет, используем price_increment как базу
                mid_estimate = float(self.instrument.price_increment) * 100000 # Примерный эстимейт для DOGE/USDT
        except Exception as e:
            # На всякий случай, если что-то пойдет не так, используем дефолтное значение
            self.log.warning(f"Could not estimate mid price: {e}. Using default.", color=LogColor.YELLOW)
            self.file_logger.warning(f"Could not estimate mid price: {e}. Using default.")
            mid_estimate = 0.1 # Разумное значение по умолчанию для DOGE/USDT
        # Оцениваем диапазон цен на основе price_increment и средней цены
        # Используем ±20% от средней цены как пределы для бинов
        estimated_min = max(float(self.instrument.price_increment), mid_estimate * 0.8) # Минимум не ниже price_increment
        estimated_max = mid_estimate * 1.2 # Максимум на 20% выше
        # Убедимся, что estimated_min < estimated_max
        if estimated_min >= estimated_max:
             estimated_max = estimated_min * 2
        self.fixed_bins = np.logspace(np.log10(estimated_min), np.log10(estimated_max), num=50)
        self.log.info(f"Fixed logarithmic bins created for volume aggregation (num={len(self.fixed_bins)}).", color=LogColor.CYAN)
        self.file_logger.info(f"Fixed logarithmic bins created for volume aggregation (num={len(self.fixed_bins)}).")
    def _add_tick_to_buffer(self, tick: QuoteTick):
        """
        Добавляет один тик в numpy-буфер с круговым доступом.
        """
        # Преобразуем время в pandas Timestamp
        ts = pd.Timestamp(tick.ts_init)
        # Получаем цены и объемы
        bid = tick.bid_price.as_double()
        ask = tick.ask_price.as_double()
        mid = (bid + ask) / 2
        # Рассчитываем спред в процентах
        spread_pct = (ask - bid) / mid if mid > 0 else 0
        # Записываем данные в буфер по текущему индексу
        self.ticks_buffer[self.buffer_idx] = (
            ts, bid, ask,
            tick.bid_size.as_double(), tick.ask_size.as_double(),
            mid, spread_pct
        )
        # Обновляем индекс, создавая эффект кольцевого буфера
        self.buffer_idx = (self.buffer_idx + 1) % MAX_BUFFER_SIZE
        # Если индекс вернулся к 0, значит буфер был полностью заполнен
        if self.buffer_idx == 0:
            self.buffer_filled = True
    def _get_recent_ticks(self, n: int) -> np.ndarray:
        """
        Возвращает последние n тиков из буфера, учитывая его кольцевую структуру.
        """
        # Если буфер еще не заполнен и в нем меньше n элементов, возвращаем все
        if not self.buffer_filled and self.buffer_idx < n:
            return self.ticks_buffer[:self.buffer_idx]
        # Рассчитываем начальный индекс для извлечения n элементов
        start = (self.buffer_idx - n) % MAX_BUFFER_SIZE
        # Если данные не пересекают границу буфера, просто возвращаем срез
        if start + n <= MAX_BUFFER_SIZE:
            return self.ticks_buffer[start:start+n]
        else:
            # Если данные пересекают границу, соединяем две части
            part1 = self.ticks_buffer[start:]
            part2 = self.ticks_buffer[:n - len(part1)]
            return np.concatenate([part1, part2])
    def on_quote_tick(self, tick: QuoteTick):
        """
        Основной обработчик котировок — здесь происходит вся логика стратегии.
        """
        # Получаем время тика
        tick_time = pd.Timestamp(tick.ts_init)
        # Пропускаем дубликаты (если время не изменилось или пошло назад)
        if self.last_tick_time is not None and tick_time <= self.last_tick_time:
            return
        # Обновляем время последнего тика
        self.last_tick_time = tick_time
        self.profiling['total_ticks'] += 1
        # Добавляем тик в буфер
        self._add_tick_to_buffer(tick)
        # --- Проверка спреда в процентах ---
        # Рассчитываем текущую серединную цену и спред
        mid = (tick.bid_price.as_double() + tick.ask_price.as_double()) / 2
        spread_pct = (tick.ask_price.as_double() - tick.bid_price.as_double()) / mid if mid > 0 else 0
        # Если спред слишком широкий (в 3 раза больше порога), пропускаем торговлю
        if spread_pct > self.min_spread_pct * 3:
            self.log.debug(f"Spread too wide: {spread_pct*10000:.3f} bps (threshold: {self.min_spread_pct*10000:.3f} bps). Skipping trade.", color=LogColor.YELLOW)
            self.file_logger.debug(f"Spread too wide: {spread_pct*10000:.3f} bps (threshold: {self.min_spread_pct*10000:.3f} bps). Skipping trade.")
            return
        # --- Агрегация объемов по 5 и 15 минутам ---
        # Измеряем время выполнения агрегации
        start_agg = time.perf_counter()
        self._aggregate_volume_by_time()
        self.profiling['aggregate_time'] += time.perf_counter() - start_agg
        # --- Проверка текущих уровней стакана ---
        # Измеряем время выполнения анализа
        start_analyze = time.perf_counter()
        self._analyze_order_book_and_trade(tick)
        self.profiling['analyze_time'] += time.perf_counter() - start_analyze
        # --- Обновление трейлинг-стопа и тейк-профита при движении цены ---
        # Измеряем время выполнения обновления стопов
        start_update = time.perf_counter()
        self._update_trailing_stop_and_tp(mid)
        self.profiling['update_tp_time'] += time.perf_counter() - start_update
        # --- Отмена просроченных ордеров ---
        self._cancel_stale_orders(tick_time)
        # --- Визуализация превью (раз в 5 тиков или при наличии позиции/широкого спреда) ---
        should_print = (
            self.current_position != 0 or  # Показываем превью, если есть позиция
            spread_pct > self.min_spread_pct * 2  # Или если спред стал слишком широким
        )
        if should_print and self.profiling['total_ticks'] % 5 == 0:
            self._print_preview(tick)
        # --- Профилирование: предупреждение о медленных операциях ---
        # Если любая операция заняла больше 0.01 секунды, выводим предупреждение
        if self.profiling['aggregate_time'] > 0.01 or self.profiling['analyze_time'] > 0.01:
            self.log.warning(f"Performance warning: aggregate={self.profiling['aggregate_time']:.4f}s, analyze={self.profiling['analyze_time']:.4f}s")
        # --- 🔥 ОПТИМИЗАЦИЯ: Перезагрузка буфера раз в 30 минут ---
        # Это помогает избежать проблем с переполнением и утечками памяти
        if self.buffer_filled and self.last_tick_time is not None:
            time_since_start = (pd.Timestamp(tick.ts_init) - self.last_tick_time)
            if time_since_start.total_seconds() > 30 * 60:  # 30 минут
                # Сохраняем последние 1000 тиков и сдвигаем их в начало буфера
                last_1k = self._get_recent_ticks(1000)
                self.ticks_buffer[:len(last_1k)] = last_1k
                self.buffer_idx = len(last_1k)
                self.buffer_filled = False
                self.log.debug("Buffer reset after 30 minutes of activity.", color=LogColor.YELLOW)
    def _aggregate_volume_by_time(self):
        """
        Агрегирует объемы по bid/ask за последние 5 и 15 минут.
        Создает временные уровни поддержки/сопротивления.
        """
        # Если буфер пуст, ничего не делаем
        if self.buffer_idx == 0 and not self.buffer_filled:
            return
        # Получаем все тики из буфера
        recent = self._get_recent_ticks(MAX_BUFFER_SIZE)
        # Создаем DataFrame для удобной работы с данными
        df = pd.DataFrame(recent)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        # Определяем временные границы для 5 и 15 минут
        now = df.index[-1]
        five_min_ago = now - pd.Timedelta(seconds=self.lookback_5m)
        fifteen_min_ago = now - pd.Timedelta(seconds=self.lookback_15m)
        # Фильтруем данные за последние 5 и 15 минут
        df_5m = df.loc[five_min_ago:now]
        df_15m = df.loc[fifteen_min_ago:now]
        def aggregate_by_price_levels(df, price_col, qty_col):
            """
            Вспомогательная функция для агрегации объемов по ценовым уровням.
            """
            if len(df) == 0:
                return [], []
            prices = df[price_col].values
            volumes = df[qty_col].values
            if len(prices) == 0:
                return [], []
            # 🔥 ИСПОЛЬЗУЕМ ПРЕДВАРИТЕЛЬНО ВЫЧИСЛЕННЫЕ ФИКСИРОВАННЫЕ БИНЫ!
            # Это ключевая оптимизация: мы не пересчитываем бины каждый раз
            hist, _ = np.histogram(prices, bins=self.fixed_bins, weights=volumes)
            # Вычисляем середины бинов как уровни цен
            mid_points = (self.fixed_bins[:-1] + self.fixed_bins[1:]) / 2
            return mid_points, hist
        # Агрегируем отдельно bid и ask уровни для 5 и 15 минут
        bid_levels_5m, bid_vols_5m = aggregate_by_price_levels(df_5m, 'bid', 'bid_qty')
        ask_levels_5m, ask_vols_5m = aggregate_by_price_levels(df_5m, 'ask', 'ask_qty')
        bid_levels_15m, bid_vols_15m = aggregate_by_price_levels(df_15m, 'bid', 'bid_qty')
        ask_levels_15m, ask_vols_15m = aggregate_by_price_levels(df_15m, 'ask', 'ask_qty')
        # Сохраняем агрегированные данные
        self.aggregated_5m = {
            'bid_levels': bid_levels_5m,
            'bid_volumes': bid_vols_5m,
            'ask_levels': ask_levels_5m,
            'ask_volumes': ask_vols_5m
        }
        self.aggregated_15m = {
            'bid_levels': bid_levels_15m,
            'bid_volumes': bid_vols_15m,
            'ask_levels': ask_levels_15m,
            'ask_volumes': ask_vols_15m
        }
        # Логируем самые значимые уровни (для отладки)
        if self.aggregated_5m and len(self.aggregated_5m['bid_levels']) > 0:
            top_bid_idx = np.argmax(self.aggregated_5m['bid_volumes']) if len(self.aggregated_5m['bid_volumes']) > 0 else -1
            top_ask_idx = np.argmax(self.aggregated_5m['ask_volumes']) if len(self.aggregated_5m['ask_volumes']) > 0 else -1
            if top_bid_idx >= 0:
                self.log.debug(
                    f"[5m] Top Bid Level: {self.aggregated_5m['bid_levels'][top_bid_idx]:.5f} "
                    f"(Vol: {self.aggregated_5m['bid_volumes'][top_bid_idx]:.0f})"
                )
            if top_ask_idx >= 0:
                self.log.debug(
                    f"[5m] Top Ask Level: {self.aggregated_5m['ask_levels'][top_ask_idx]:.5f} "
                    f"(Vol: {self.aggregated_5m['ask_volumes'][top_ask_idx]:.0f})"
                )
    def _analyze_order_book_and_trade(self, latest_tick: QuoteTick):
        """
        Анализирует стакан и принимает решение о выставлении лимитных ордеров.
        """
        # Если уже есть открытая позиция — не открываем новые
        if self.current_position != 0:
            return
        # Проверяем, есть ли уже ожидающие ордера — если да, пропускаем
        if len(self.pending_orders) > 0:
            self.log.debug(f"Pending orders exist: {len(self.pending_orders)}. Skipping new order.", color=LogColor.YELLOW)
            return
        # Проверяем cooldown между входами
        if self.last_trade_time and (pd.Timestamp(latest_tick.ts_init) - self.last_trade_time) < pd.Timedelta(seconds=self.trade_cooldown_seconds):
            self.log.debug(f"Trade cooldown active ({self.trade_cooldown_seconds}s). Skipping.", color=LogColor.YELLOW)
            return
        # Проверяем коулдаун после закрытия позиции
        if self.last_close_time and (pd.Timestamp(latest_tick.ts_init) - self.last_close_time) < pd.Timedelta(seconds=self.post_trade_cooldown_seconds):
            self.log.debug(f"Post-trade cooldown active ({self.post_trade_cooldown_seconds}s). Skipping.", color=LogColor.YELLOW)
            return
        # Получаем лучшие цены bid и ask
        best_bid = latest_tick.bid_price.as_double()
        best_ask = latest_tick.ask_price.as_double()
        mid = (best_bid + best_ask) / 2
        # Проверяем, что стакан не пуст
        if best_bid == 0 or best_ask == 0:
            self.log.debug("Market illiquid: bid or ask is zero. Waiting...", color=LogColor.YELLOW)
            return
        # --- 🔥 ОПТИМИЗАЦИЯ: Фильтр минимального общего объема рынка ---
        # Проверяем, достаточно ли объема на рынке за последние 5 секунд
        recent = self._get_recent_ticks(100)  # ~100 тиков за 5 секунд при 20 тиках/сек
        if len(recent) < 50:  # Не хватает данных
            return
        total_bid_vol_5s = np.sum(recent['bid_qty'])
        total_ask_vol_5s = np.sum(recent['ask_qty'])
        min_total_volume_threshold = 500  # Например, 500 единиц базовой валюты за 5 секунд
        if total_bid_vol_5s + total_ask_vol_5s < min_total_volume_threshold:
            self.log.debug(f"Low market volume: {total_bid_vol_5s + total_ask_vol_5s:.0f} < {min_total_volume_threshold}. Skipping trade.", color=LogColor.YELLOW)
            return
        # --- Проверка доступного капитала ---
        # Получаем информацию о счете
        # Исправление: используем venue из instrument_id для получения account_id
        # account = self.cache.account(self.instrument_id.account_id) # ❌ Старая строка с ошибкой
        # account = self.cache.account(self.instrument_id.venue) # ❌ Старая строка с ошибкой типа
        # Исправление: Создаем правильный AccountId
        # from nautilus_trader.model.identifiers import AccountId
        account_id = AccountId(f"{self.instrument_id.venue.value}-UNIFIED") # ✅ Исправленная строка
        account = self.cache.account(account_id)
        if not account:
            self.log.warning("Account not found. Skipping order.")
            return
        # Получаем общий баланс
        # Исправление: правильный способ получения баланса
        # balance_total = account.balance_total(Money).as_decimal() # ❌ Старая строка с ошибкой
        # balance_obj = account.balance_total(None) # ❌ Старая строка с ошибкой 'currency' argument was `None`
        # Исправление: Создаем объект Currency для USDT
        # from nautilus_trader.model.objects import Currency
        usdt_currency = Currency.from_str("USDT") # ✅ Создаем валюту USDT
        balance_obj = account.balance_total(usdt_currency) # ✅ Передаем валюту в метод
        # if not balance_obj: # ❌ Старая проверка
        #      self.log.warning("Could not retrieve total balance. Skipping order.")
        #      return
        # balance_total = balance_obj.total.as_decimal() # ❌ Старая строка с ошибкой 'Money' object has no attribute 'total'
        if not isinstance(balance_obj, Money): # ✅ Проверяем, что объект Money возвращен
             self.log.warning("Could not retrieve total balance. Skipping order.")
             return
        balance_total = balance_obj.as_decimal() # ✅ Исправленная строка
        # Рассчитываем требуемый маржин (цена * размер + запас на комиссии)
        required_margin = Decimal(str(mid)) * self.trade_size
        if balance_total < required_margin * Decimal("1.01"):  # +1% на комиссии
            self.log.warning(f"Not enough balance: {balance_total:.2f} < {required_margin:.2f}. Skipping order.", color=LogColor.RED)
            return
        # --- Определение тренда по последним 5 тикам ---
        recent = self._get_recent_ticks(5)
        if len(recent) < 5:
            return
        recent_bids = recent['bid']
        recent_asks = recent['ask']
        # Рассчитываем простые скользящие средние
        sma5_bid = np.mean(recent_bids[-5:])
        sma10_bid = np.mean(recent_bids[-10:] if len(recent_bids) >= 10 else recent_bids)
        # Определяем тренд для bid: рост, если SMA5 > SMA10 и разница значительна
        trend_bid = sma5_bid > sma10_bid and (sma5_bid - sma10_bid) / sma10_bid > 0.0002 if sma10_bid > 0 else False
        sma5_ask = np.mean(recent_asks[-5:])
        sma10_ask = np.mean(recent_asks[-10:] if len(recent_asks) >= 10 else recent_asks)
        # Определяем тренд для ask: падение, если SMA5 < SMA10 и разница значительна
        trend_ask = sma5_ask < sma10_ask and (sma10_ask - sma5_ask) / sma10_ask > 0.0002 if sma10_ask > 0 else False
        is_up_trend = trend_bid
        is_down_trend = trend_ask
        # --- Поиск уровней поддержки/сопротивления ---
        bid_support_levels = []  # Уровни поддержки (много bid объема)
        ask_resistance_levels = []  # Уровни сопротивления (много ask объема)
        # Анализируем агрегированные данные за 5 минут
        if self.aggregated_5m:
            # Собираем уровни поддержки с достаточным объемом
            for level, vol in zip(self.aggregated_5m['bid_levels'], self.aggregated_5m['bid_volumes']):
                if vol >= self.min_volume_threshold:
                    bid_support_levels.append((level, vol))
            # Собираем уровни сопротивления с достаточным объемом
            for level, vol in zip(self.aggregated_5m['ask_levels'], self.aggregated_5m['ask_volumes']):
                if vol >= self.min_volume_threshold:
                    ask_resistance_levels.append((level, vol))
        # Сортируем уровни по близости к текущей средней цене (ближайшие первыми)
        bid_support_levels.sort(key=lambda x: abs(x[0] - mid), reverse=False)
        ask_resistance_levels.sort(key=lambda x: abs(x[0] - mid), reverse=False)
        # --- Логика входа ---
        # Если растет — ищем поддержку (BUY на уровне поддержки)
        if is_up_trend and len(bid_support_levels) > 0:
            support_level = None
            # Ищем уровень поддержки ниже текущей цены
            for level, vol in bid_support_levels:
                if level < mid:
                    support_level = level
                    break
            # Если такой уровень найден
            if support_level:
                # Проверяем, не слишком ли далеко он от текущей цены
                distance_to_support = mid - support_level
                max_distance = max(self.instrument.tick_size * 2, mid * self.max_entry_distance_factor)
                if distance_to_support < max_distance:
                    # Рассчитываем потенциальную прибыль
                    target_exit_price = support_level * (1 + self.take_profit_pct)
                    profit_potential_pct = (target_exit_price - support_level) / support_level
                    # Проверяем, достаточно ли потенциальной прибыли с учетом комиссий
                    if profit_potential_pct >= self.min_profit_before_commission:
                        self._place_limit_order(OrderSide.BUY, support_level, "UP_TREND_SUPPORT")
                    else:
                        self.log.debug(f"Buy signal at {support_level:.5f}: profit potential {profit_potential_pct*100:.5f}% < min {self.min_profit_before_commission*100:.5f}%. Skipping.", color=LogColor.YELLOW)
        # Если падает — ищем сопротивление (SELL на уровне сопротивления)
        elif is_down_trend and len(ask_resistance_levels) > 0:
            resistance_level = None
            # Ищем уровень сопротивления выше текущей цены
            for level, vol in ask_resistance_levels:
                if level > mid:
                    resistance_level = level
                    break
            # Если такой уровень найден
            if resistance_level:
                # Проверяем, не слишком ли далеко он от текущей цены
                distance_to_resistance = resistance_level - mid
                max_distance = max(self.instrument.tick_size * 2, mid * self.max_entry_distance_factor)
                if distance_to_resistance < max_distance:
                    # Рассчитываем потенциальную прибыль
                    target_exit_price = resistance_level * (1 - self.take_profit_pct)
                    profit_potential_pct = (resistance_level - target_exit_price) / resistance_level
                    # Проверяем, достаточно ли потенциальной прибыли с учетом комиссий
                    if profit_potential_pct >= self.min_profit_before_commission:
                        self._place_limit_order(OrderSide.SELL, resistance_level, "DOWN_TREND_RESISTANCE")
                    else:
                        self.log.debug(f"Sell signal at {resistance_level:.5f}: profit potential {profit_potential_pct*100:.5f}% < min {self.min_profit_before_commission*100:.5f}%. Skipping.", color=LogColor.YELLOW)
    def _place_limit_order(self, side: OrderSide, price: float, reason: str):
        """
        Выставляет лимитный ордер. Проверяет наличие pending ордеров.
        """
        # Генерируем уникальный ID для ордера
        client_order_id = ClientOrderId(f"SCALP_{side.name}_{self.order_id_counter}_{price:.5f}")
        # Преобразуем размер сделки в объект Quantity
        quantity = Quantity.from_str(str(self.trade_size))
        # Создаем лимитный ордер
        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=quantity,
            price=Price.from_str(str(price)),
            time_in_force=TimeInForce.GTC, # Good-Till-Cancelled
            client_order_id=client_order_id,
        )
        # Логируем отправку ордера
        self.log.info(f"Submitting LIMIT {side.name} order: {price:.5f} | Qty: {quantity} | Reason: {reason}", color=LogColor.CYAN)
        self.file_logger.info(f"Submitting LIMIT {side.name} order: {price:.5f} | Qty: {quantity} | Reason: {reason}")
        # Отправляем ордер в брокер
        self.submit_order(order)
        # Добавляем ордер в список ожидания и сохраняем время выставления
        self.pending_orders[client_order_id] = self.last_tick_time
        self.order_id_counter += 1
        # Фиксируем время входа для коулдауна
        self.last_trade_time = self.last_tick_time
        # Обновляем статистику
        self.profiling['submitted_orders'] += 1
    def _update_trailing_stop_and_tp(self, current_mid: float):
        """
        Обновляет трейлинг-стоп и тейк-профит при изменении цены.
        Работает только при наличии открытой позиции.
        """
        # Если позиции нет, ничего не делаем
        if self.current_position == 0:
            return
        # Расчет расстояний для стопов и тейков в абсолютных значениях
        trailing_stop_distance = current_mid * self.trailing_stop_pct
        take_profit_distance = self.entry_price * self.take_profit_pct # Используем цену входа для TP
        # --- Исправленный блок ---
        if self.current_position == 1:  # LONG
            # Если это первый вход в лонг, инициализируем уровни
            if self.entry_price == 0:
                self.entry_price = current_mid
                self.take_profit_price = self.entry_price + take_profit_distance
                self.trailing_stop_price = current_mid - trailing_stop_distance
                self.log.info(f"✅ LONG opened at {self.entry_price:.5f}. TP: {self.take_profit_price:.5f}, TS: {self.trailing_stop_price:.5f}", color=LogColor.GREEN)
                self.file_logger.info(f"✅ LONG opened at {self.entry_price:.5f}. TP: {self.take_profit_price:.5f}, TS: {self.trailing_stop_price:.5f}")
                return # Выходим, так как инициализация завершена
            # Обновляем трейлинг-стоп вверх — только если цена поднялась достаточно
            potential_new_ts = current_mid - trailing_stop_distance
            if potential_new_ts > self.trailing_stop_price: # TS следует за ценой вверх
                 self.trailing_stop_price = potential_new_ts
            # Проверяем тейк-профит
            if current_mid >= self.take_profit_price:
                self.log.info(f"✅ TAKE PROFIT HIT! LONG closed at {current_mid:.5f}", color=LogColor.GREEN)
                self.file_logger.info(f"✅ TAKE PROFIT HIT! LONG closed at {current_mid:.5f}")
                self.close_position()
                return
            # Проверяем трейлинг-стоп
            if current_mid <= self.trailing_stop_price: # Цена упала до уровня TS
                self.log.info(f"⛔ TRAILING STOP HIT! LONG closed at {current_mid:.5f}", color=LogColor.RED)
                self.file_logger.info(f"⛔ TRAILING STOP HIT! LONG closed at {current_mid:.5f}")
                self.close_position()
                return
        elif self.current_position == -1:  # SHORT
            # Если это первый вход в шорт, инициализируем уровни
            if self.entry_price == 0:
                self.entry_price = current_mid
                self.take_profit_price = self.entry_price - take_profit_distance
                self.trailing_stop_price = current_mid + trailing_stop_distance
                self.log.info(f"❌ SHORT opened at {self.entry_price:.5f}. TP: {self.take_profit_price:.5f}, TS: {self.trailing_stop_price:.5f}", color=LogColor.YELLOW)
                self.file_logger.info(f"❌ SHORT opened at {self.entry_price:.5f}. TP: {self.take_profit_price:.5f}, TS: {self.trailing_stop_price:.5f}")
                return # Выходим, так как инициализация завершена
            # Обновляем трейлинг-стоп вниз — только если цена опустилась достаточно
            potential_new_ts = current_mid + trailing_stop_distance
            if potential_new_ts < self.trailing_stop_price: # TS следует за ценой вниз
                 self.trailing_stop_price = potential_new_ts
             # Проверяем тейк-профит
            if current_mid <= self.take_profit_price:
                self.log.info(f"✅ TAKE PROFIT HIT! SHORT closed at {current_mid:.5f}", color=LogColor.GREEN)
                self.file_logger.info(f"✅ TAKE PROFIT HIT! SHORT closed at {current_mid:.5f}")
                self.close_position()
                return
            # Проверяем трейлинг-стоп
            if current_mid >= self.trailing_stop_price: # Цена поднялась до уровня TS
                self.log.info(f"⛔ TRAILING STOP HIT! SHORT closed at {current_mid:.5f}", color=LogColor.RED)
                self.file_logger.info(f"⛔ TRAILING STOP HIT! SHORT closed at {current_mid:.5f}")
                self.close_position()
                return
    def close_position(self):
        """
        Закрывает текущую позицию рыночным ордером.
        """
        # Если позиции нет, ничего не делаем
        if self.current_position == 0:
            return
        # Определяем сторону рыночного ордера для закрытия позиции
        side = OrderSide.SELL if self.current_position == 1 else OrderSide.BUY
        # Преобразуем размер сделки в объект Quantity
        quantity = Quantity.from_str(str(self.trade_size))
        # Создаем рыночный ордер
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=quantity,
            client_order_id=ClientOrderId(f"CLOSE_{self.order_id_counter}"),
        )
        # Логируем отправку ордера
        self.log.info(f"Closing position with MARKET order: {side.name} {quantity}", color=LogColor.MAGENTA)
        self.file_logger.info(f"Closing position with MARKET order: {side.name} {quantity}")
        # Отправляем ордер в брокер
        self.submit_order(order)
        # Фиксируем время закрытия для коулдауна
        self.last_close_time = self.last_tick_time
        # Сбрасываем состояние стратегии
        self.current_position = 0
        self.entry_price = 0.0
        self.trailing_stop_price = 0.0
        self.take_profit_price = 0.0
        self.position_size = Decimal("0")
        self.order_id_counter += 1
    def _cancel_stale_orders(self, current_time: pd.Timestamp):
        """
        Отменяет ордера, которые не исполнились за MAX_ORDER_AGE_SECONDS.
        """
        # Список для хранения ID ордеров, которые нужно отменить
        to_cancel = []
        # Проходим по всем ожидающим ордерам
        for oid, placed_at in self.pending_orders.items():
            # Если ордер висит дольше, чем разрешено
            if (current_time - placed_at) > pd.Timedelta(seconds=self.max_order_age_seconds):
                to_cancel.append(oid)
        # Отменяем просроченные ордера
        for oid in to_cancel:
            self.cancel_order(oid)
            # Удаляем из списка ожидания
            del self.pending_orders[oid]
            self.log.info(f"Canceled stale limit order: {oid}", color=LogColor.YELLOW)
            self.file_logger.info(f"Canceled stale limit order: {oid}")
            self.profiling['canceled_orders'] += 1
    def _print_preview(self, latest_tick: QuoteTick):
        """
        Выводит красивое превью стакана и ключевых уровней с помощью tabulate.
        """
        if not self.instrument:
            return
        # Формируем таблицу стакана
        levels = []
        for i in range(1, self.order_book_depth + 1):
            levels.append([
                f"Ask {i}",
                f"{latest_tick.ask_price.as_double():.5f}",
                f"{latest_tick.ask_size.as_double():.2f}",
                f"Bid {i}",
                f"{latest_tick.bid_price.as_double():.5f}",
                f"{latest_tick.bid_size.as_double():.2f}"
            ])
        # Добавляем агрегированные уровни
        agg_levels = []
        if self.aggregated_5m and len(self.aggregated_5m['bid_levels']) > 0:
            # Находим уровень с максимальным объемом bid за 5 минут
            if len(self.aggregated_5m['bid_volumes']) > 0:
                top_bid_idx = np.argmax(self.aggregated_5m['bid_volumes'])
                top_bid = (self.aggregated_5m['bid_levels'][top_bid_idx], self.aggregated_5m['bid_volumes'][top_bid_idx])
            else:
                top_bid = (0, 0)
            # Находим уровень с максимальным объемом ask за 5 минут
            if len(self.aggregated_5m['ask_volumes']) > 0:
                top_ask_idx = np.argmax(self.aggregated_5m['ask_volumes'])
                top_ask = (self.aggregated_5m['ask_levels'][top_ask_idx], self.aggregated_5m['ask_volumes'][top_ask_idx])
            else:
                top_ask = (0, 0)
            # Формируем строки для таблицы агрегированных уровней
            agg_levels = [
                ["5m Support", f"{top_bid[0]:.5f}", f"{top_bid[1]:.0f}"],
                ["5m Resistance", f"{top_ask[0]:.5f}", f"{top_ask[1]:.0f}"]
            ]
        # Таблица стакана
        headers = ["ASK Level", "Price", "Volume", "BID Level", "Price", "Volume"]
        table = tabulate(levels, headers=headers, tablefmt="grid", floatfmt=".5f")
        # Таблица уровней
        agg_headers = ["Level Type", "Price", "Volume"]
        agg_table = tabulate(agg_levels, headers=agg_headers, tablefmt="simple", floatfmt=".5f")
        # Тренд
        recent = self._get_recent_ticks(5)
        prices = recent['mid']
        trend = "UP" if prices[-1] > prices[0] else "DOWN" if prices[-1] < prices[0] else "FLAT"
        # Спред в базовых пунктах (bps)
        spread_bps = (latest_tick.ask_price.as_double() - latest_tick.bid_price.as_double()) / ((latest_tick.bid_price.as_double() + latest_tick.ask_price.as_double()) / 2) * 10000
        # Метрики производительности
        submitted = self.profiling['submitted_orders']
        filled = self.profiling['filled_orders']
        canceled = self.profiling['canceled_orders']
        fill_rate = (filled / submitted * 100) if submitted > 0 else 0
        # Общий превью
        preview = f"""
{'='*80}
📊 SCALPING PREVIEW ({self.instrument.symbol}) — PROPORTIONAL MODE WITH COMMISSION (0.04%)
{'='*80}
🕒 Latest Tick: {pd.Timestamp(latest_tick.ts_init):%H:%M:%S.%f}
💰 Mid Price: {latest_tick.mid:.5f} | Spread: {spread_bps:.2f} bps ({self.min_spread_pct*10000:.2f} bps threshold)
📈 Trend (last 5 ticks): {trend}
📉 Current Position: {self._get_position_string()}
{table}
📌 AGGREGATED LEVELS (5m):
{agg_table}
ℹ️ Trailing Stop: {self.trailing_stop_price:.5f} ({self.trailing_stop_pct*100:.4f}% of price)
ℹ️ Take Profit: {self.take_profit_price:.5f} ({self.take_profit_pct*100:.4f}% of entry)
ℹ️ Total PnL: {float(self.total_pnl):.5f} | Max Daily Loss: {self.max_daily_loss_pct*100:.2f}% (Capital: {float(self.initial_capital):.2f})
ℹ️ Pending Orders: {len(self.pending_orders)} | Last Trade: {self.last_trade_time}
ℹ️ Last Close: {self.last_close_time} | Post-cooldown: {self.post_trade_cooldown_seconds}s
📊 Performance: Filled: {filled}/{submitted} ({fill_rate:.1f}%) | Canceled: {canceled}
⏱️ Profiling: Aggregate={self.profiling['aggregate_time']:.4f}s | Analyze={self.profiling['analyze_time']:.4f}s | UpdateTP={self.profiling['update_tp_time']:.4f}s
{'='*80}
        """.strip()
        # 🔥 ОПТИМИЗАЦИЯ: Логируем ТОЛЬКО в консоль, чтобы не засорять файл
        self.log.info(preview, color=LogColor.BLUE)
        # self.file_logger.info(preview)  # 👉 УБРАНО: файловый лог больше не пишется для превью
    def on_order_filled(self, order_filled):
        """
        Обрабатывает исполнение ордера — обновляет позицию и PnL.
        """
        self.log.info(f"Order filled: {order_filled}", color=LogColor.MAGENTA)
        self.file_logger.info(f"Order filled: {order_filled}")
        self.profiling['filled_orders'] += 1
        # Убираем исполненный ордер из списка ожидания
        if order_filled.client_order_id in self.pending_orders:
            del self.pending_orders[order_filled.client_order_id]
        # Получаем объект ордера из кэша
        order = self.cache.order(order_filled.client_order_id)
        if not order:
            return
        # Получаем цену и количество исполнения
        fill_price = order_filled.last_px.as_double()
        fill_qty = order_filled.last_qty.as_double()
        # 🔴 Обработка частичных исполнений — пересчёт средней цены
        # Проверяем, совпадает ли исполненное количество с заявленным
        if abs(fill_qty - self.trade_size.as_decimal()) > 1e-8:
            self.log.warning(f"Partial fill detected: {fill_qty} vs trade_size {self.trade_size.as_decimal()}. Updating average entry.", color=LogColor.YELLOW)
            self.file_logger.warning(f"Partial fill detected: {fill_qty} vs trade_size {self.trade_size.as_decimal()}. Updating average entry.")
        # Обработка исполнения BUY ордера
        if order.side == OrderSide.BUY:
            # Если это открытие новой позиции (лонг)
            if self.current_position == 0:
                self.current_position = 1
                self.entry_price = fill_price
                self.position_size = Decimal(str(fill_qty))
                # Устанавливаем уровни TP и TS
                self.take_profit_price = self.entry_price * (1 + self.take_profit_pct)
                self.trailing_stop_price = self.entry_price * (1 - self.trailing_stop_pct)
                self.log.info(f"✅ LONG opened at {fill_price:.5f} | Size: {fill_qty:.6f}", color=LogColor.GREEN)
                self.file_logger.info(f"✅ LONG opened at {fill_price:.5f} | Size: {fill_qty:.6f}")
            # Если это закрытие позиции (выход из шорта)
            elif self.current_position == -1:
                # Рассчитываем реализованный PnL
                realized_pnl = (self.entry_price - fill_price) * fill_qty
                self.total_pnl += Decimal(str(realized_pnl))
                # Уменьшаем размер позиции
                self.position_size -= Decimal(str(fill_qty))
                # Если позиция полностью закрыта
                if abs(self.position_size) < 1e-8:
                    self.current_position = 0
                    self.entry_price = 0.0
                    self.log.info(f"🔁 SHORT closed at {fill_price:.5f} | PnL: {float(realized_pnl):.5f}", color=LogColor.CYAN)
                    self.file_logger.info(f"🔁 SHORT closed at {fill_price:.5f} | PnL: {float(realized_pnl):.5f}")
                else:
                    # Если позиция закрыта частично, пересчитываем среднюю цену входа
                    self.entry_price = (self.entry_price * (self.trade_size.as_decimal() - Decimal(str(fill_qty))) + fill_price * Decimal(str(fill_qty))) / self.position_size
                    # Обновляем уровни TP и TS
                    self.take_profit_price = self.entry_price * (1 + self.take_profit_pct)
                    self.trailing_stop_price = self.entry_price * (1 - self.trailing_stop_pct)
                    self.log.info(f"🔁 SHORT partially closed at {fill_price:.5f} | Remaining: {self.position_size:.6f}", color=LogColor.CYAN)
                    self.file_logger.info(f"🔁 SHORT partially closed at {fill_price:.5f} | Remaining: {self.position_size:.6f}")
        # Обработка исполнения SELL ордера
        elif order.side == OrderSide.SELL:
            # Если это открытие новой позиции (шорт)
            if self.current_position == 0:
                self.current_position = -1
                self.entry_price = fill_price
                self.position_size = Decimal(str(fill_qty))
                # Устанавливаем уровни TP и TS
                self.take_profit_price = self.entry_price * (1 - self.take_profit_pct)
                self.trailing_stop_price = self.entry_price * (1 + self.trailing_stop_pct)
                self.log.info(f"❌ SHORT opened at {fill_price:.5f} | Size: {fill_qty:.6f}", color=LogColor.YELLOW)
                self.file_logger.info(f"❌ SHORT opened at {fill_price:.5f} | Size: {fill_qty:.6f}")
            # Если это закрытие позиции (выход из лонга)
            elif self.current_position == 1:
                # Рассчитываем реализованный PnL
                realized_pnl = (fill_price - self.entry_price) * fill_qty
                self.total_pnl += Decimal(str(realized_pnl))
                # Уменьшаем размер позиции
                self.position_size -= Decimal(str(fill_qty))
                # Если позиция полностью закрыта
                if abs(self.position_size) < 1e-8:
                    self.current_position = 0
                    self.entry_price = 0.0
                    self.log.info(f"🔁 LONG closed at {fill_price:.5f} | PnL: {float(realized_pnl):.5f}", color=LogColor.CYAN)
                    self.file_logger.info(f"🔁 LONG closed at {fill_price:.5f} | PnL: {float(realized_pnl):.5f}")
                else:
                    # Если позиция закрыта частично, пересчитываем среднюю цену входа
                    self.entry_price = (self.entry_price * (self.trade_size.as_decimal() - Decimal(str(fill_qty))) + fill_price * Decimal(str(fill_qty))) / self.position_size
                    # Обновляем уровни TP и TS
                    self.take_profit_price = self.entry_price * (1 + self.take_profit_pct)
                    self.trailing_stop_price = self.entry_price * (1 - self.trailing_stop_pct)
                    self.log.info(f"🔁 LONG partially closed at {fill_price:.5f} | Remaining: {self.position_size:.6f}", color=LogColor.CYAN)
                    self.file_logger.info(f"🔁 LONG partially closed at {fill_price:.5f} | Remaining: {self.position_size:.6f}")
        # 🔴 КРИТИЧЕСКОЕ: Проверка на достижение лимита просадки по КАПИТАЛУ
        # Рассчитываем порог просадки
        daily_loss_threshold = self.initial_capital * self.max_daily_loss_pct
        # Если общий PnL упал ниже порога, останавливаем стратегию
        if self.total_pnl < -daily_loss_threshold:
            self.log.critical(f"⚠️ MAX DAILY LOSS REACHED: {float(self.total_pnl):.5f} < {-float(daily_loss_threshold):.5f}. Stopping strategy.", color=LogColor.RED)
            self.file_logger.critical(f"MAX DAILY LOSS REACHED: {float(self.total_pnl):.5f} < {-float(daily_loss_threshold):.5f}. Stopping strategy.")
            self.stop()
    def on_order_canceled(self, order_canceled):
        """
        Убираем ордер из pending, если он отменён.
        """
        # Удаляем отмененный ордер из списка ожидания
        if order_canceled.client_order_id in self.pending_orders:
            del self.pending_orders[order_canceled.client_order_id]
            self.log.debug(f"Order canceled: {order_canceled.client_order_id}", color=LogColor.YELLOW)
            self.file_logger.debug(f"Order canceled: {order_canceled.client_order_id}")
            self.profiling['canceled_orders'] += 1
    def on_order_updated(self, order_updated):
        """
        При обновлении ордера — ничего не делаем.
        """
        pass
    def on_stop(self):
        """
        Вызывается при остановке стратегии.
        """
        # Логируем финальное состояние
        self.log.info(f"ScalpStrategy stopped. Final state: {self._get_position_string()} | Pending orders: {len(self.pending_orders)} | Total PnL: {float(self.total_pnl):.5f}", color=LogColor.MAGENTA)
        self.file_logger.info(f"ScalpStrategy stopped. Final state: {self._get_position_string()} | Pending orders: {len(self.pending_orders)} | Total PnL: {float(self.total_pnl):.5f}")
    def on_reset(self):
        """
        Сброс состояния стратегии.
        """
        self.log.info("ScalpStrategy reset initiated.", color=LogColor.MAGENTA)
        self.file_logger.info("ScalpStrategy reset initiated.")
        # Сбрасываем все внутренние переменные
        self.buffer_idx = 0
        self.buffer_filled = False
        self.current_position = 0
        self.entry_price = 0.0
        self.trailing_stop_price = 0.0
        self.take_profit_price = 0.0
        self.pending_orders.clear()
        self.order_id_counter = 0
        self.last_tick_time = None
        self.last_trade_time = None
        self.last_close_time = None
        self.total_pnl = Decimal("0")
        self.position_size = Decimal("0")
        # Сбрасываем профилирование
        for key in self.profiling:
            self.profiling[key] = 0
        self.log.info("ScalpStrategy reset completed.", color=LogColor.MAGENTA)
        self.file_logger.info("ScalpStrategy reset completed.")
    def on_dispose(self):
        """
        Уничтожение стратегии.
        """
        self.log.info("ScalpStrategy disposed.", color=LogColor.MAGENTA)
        self.file_logger.info("ScalpStrategy disposed.")
    def _get_position_string(self):
        """Возвращает строковое представление позиции."""
        if self.current_position == 1:
            return f"LONG ({float(self.position_size):.6f})"
        elif self.current_position == -1:
            return f"SHORT ({float(self.position_size):.6f})"
        else:
            return "FLAT"

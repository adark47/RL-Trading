# strategy.py
import torch
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy
from agent import D3QN_PER_Agent  # Импортируем вашу RL модель
from tabulate import tabulate
from decimal import Decimal

# Добавляем логгер
logger = logging.getLogger(__name__)


class RLStrategyConfig(StrategyConfig, frozen=True):
    """Конфигурация для RL-стратегии"""
    instrument_id: str
    primary_bar_type: BarType
    trade_size: Decimal
    profit_in_ticks: int
    stoploss_in_ticks: int
    model_path: str  # Путь к обученной модели
    state_window: int = 50  # Окно исторических данных для состояния
    action_dim: int = 3  # 0=hold, 1=buy, 2=sell

    # Параметры для нормализации (должны соответствовать обучению)
    price_channels: List[str] = ["open", "high", "low", "close"]
    volume_channels: List[str] = ["volume"]
    other_channels: List[str] = []
    data_channels: List[str] = ["open", "high", "low", "close", "volume"]

    # Параметры архитектуры модели (должны соответствовать обучению)
    cnn_maps: List[int] = [32, 64, 64]
    cnn_kernels: List[int] = [8, 4, 3]
    cnn_strides: List[int] = [4, 2, 1]
    dense_val: List[int] = [512]
    dense_adv: List[int] = [512]
    additional_feats: int = 4  # position, unrealized_pnl, time_elapsed, time_remaining
    dropout_model: float = 0.2
    num_features: int = 5  # open, high, low, close, volume
    input_history_len: int = 50
    agent_history_len: int = 50
    pre_signal_len: int = 0
    action_history_len: int = 10

    # Статистики для нормализации (должны соответствовать обучению)
    normalization_stats: Optional[Dict[str, Dict[str, float]]] = None


class RLTradingStrategy(Strategy):
    """
    Стратегия, использующая обученную RL-модель для принятия торговых решений
    """

    def __init__(self, config: RLStrategyConfig):
        super().__init__(config)
        # Инициализация данных
        self.dfBars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.dfBars.index.name = 'time'
        self.dfBars.index = pd.to_datetime(self.dfBars.index)

        self.current_position = None  # None, OrderSide.BUY, OrderSide.SELL
        self.entry_price = None
        self.last_action = 0  # 0=hold по умолчанию
        self.action_history = [0] * config.action_history_len  # История действий

        # Инициализация RL агента
        self._initialize_rl_agent(config)

        # Проверка совместимости
        self._validate_compatibility(config)

        # Статистики для нормализации
        self.normalization_stats = config.normalization_stats
        if not self.normalization_stats:
            logger.warning("Warning: No normalization stats provided. Model performance may be suboptimal.")

        self.log.info("RL Trading Strategy initialized successfully", color=LogColor.GREEN)

    def _validate_compatibility(self, config: RLStrategyConfig) -> None:
        """Проверка совместимости конфигурации со структурой модели"""
        expected_state_shape = (
            config.num_features,
            config.input_history_len,
            1
        )

        # Проверяем соответствие формы состояния
        if tuple(self.agent.policy_net.conv_input_shape) != expected_state_shape:
            self.log.error(
                f"State shape mismatch! Model expects {self.agent.policy_net.conv_input_shape}, "
                f"but configuration specifies {expected_state_shape}",
                color=LogColor.RED
            )
            raise ValueError("Incompatible state shape configuration")

    def _initialize_rl_agent(self, config: RLStrategyConfig) -> None:
        """Инициализация RL агента с загрузкой обученной модели"""
        try:
            # Определяем устройство
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log.info(f"Using device: {device}", color=LogColor.BLUE)

            # Проверяем существование модели
            if not os.path.exists(config.model_path):
                self.log.error(f"Model file not found: {config.model_path}", color=LogColor.RED)
                raise FileNotFoundError(f"Model file not found: {config.model_path}")

            # Создаем агента с параметрами, соответствующими обучению
            self.agent = D3QN_PER_Agent(
                state_shape=(config.num_features, config.input_history_len, 1),
                action_dim=4,  # Модель обучалась на 4 действиях (0=hold, 1=long, 2=short, 3=close)
                cnn_maps=config.cnn_maps,
                cnn_kernels=config.cnn_kernels,
                cnn_strides=config.cnn_strides,
                dense_val=config.dense_val,
                dense_adv=config.dense_adv,
                additional_feats=config.additional_feats,
                dropout_model=config.dropout_model,
                device=device,
                gamma=0.99,  # Можно вынести в конфиг
                learning_rate=1e-4,  # Не используется при инференсе
                batch_size=64,  # Не используется при инференсе
                buffer_size=50000,  # Не используется при инференсе
                target_update_freq=1000,  # Не используется при инференсе
                train_start=1000,  # Не используется при инференсе
                per_alpha=0.6,  # Не используется при инференсе
                per_beta_start=0.4,  # Не используется при инференсе
                per_beta_frames=100000,  # Не используется при инференсе
                eps_start=1.0,  # Не используется при инференсе
                eps_end=0.01,  # Не используется при инференсе
                eps_frames=50000,  # Не используется при инференсе
                epsilon=1e-6,  # Не используется при инференсе
                max_gradient_norm=40.0,  # Не используется при инференсе
                backtest_cache_path=None
            )

            # Загружаем обученные веса
            self.agent.load_model(config.model_path)
            self.log.info(f"Model loaded successfully from {config.model_path}", color=LogColor.GREEN)

            # Переводим модель в режим инференса
            self.agent.policy_net.eval()

        except Exception as e:
            self.log.error(f"Failed to initialize RL agent: {str(e)}", color=LogColor.RED)
            raise

    def on_start(self):
        """Действия при старте стратегии"""
        self.log.info("Starting RL Trading Strategy", color=LogColor.GREEN)

        # Подписываемся на бары
        self.subscribe_bars(self.config.primary_bar_type)

        # Запрашиваем исторические данные для заполнения окна
        self.request_bars(
            BarType.from_str(f"{self.config.instrument_id}-1-MINUTE-LAST-EXTERNAL")
        )

        self.log.info("Subscribed to bars and requested historical data", color=LogColor.BLUE)

    def on_bar(self, bar: Bar):
        """Обработка нового бара"""
        self.log.info(f"New bar received: {repr(bar)}", color=LogColor.CYAN)

        # Обновляем историю баров
        self._update_bars_history(bar)

        # Проверяем, достаточно ли данных для формирования состояния
        if len(self.dfBars) < self.config.state_window:
            self.log.info(
                f"Waiting for more data... Current: {len(self.dfBars)}/{self.config.state_window}",
                color=LogColor.BLUE
            )
            return

        # Формируем состояние для модели
        state = self._prepare_state()

        # Получаем действие от модели
        action = self._get_rl_action(state)

        # Выполняем торговое решение
        self._execute_trading_decision(action, bar)

        # Логируем текущее состояние
        self._log_strategy_state(bar, action)

    def _update_bars_history(self, bar: Bar):
        """Обновление истории баров"""
        # Преобразуем бар в формат DataFrame
        bar_data = {
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume)
        }

        # Добавляем в DataFrame
        timestamp = datetime.fromtimestamp(bar.ts_init / 1e9)
        self.dfBars.loc[timestamp] = bar_data

        # Ограничиваем размер истории
        if len(self.dfBars) > self.config.state_window * 2:
            self.dfBars = self.dfBars.iloc[-(self.config.state_window * 2):]

    def _prepare_state(self) -> np.ndarray:
        """Подготовка состояния для RL-модели"""
        # Выбираем последние N баров
        window = self.dfBars.iloc[-self.config.agent_history_len:].values

        # Применяем нормализацию (аналогично обучению)
        normalized = self._apply_normalization(window)

        # Формируем дополнительные признаки (аналогично TradingEnvironment)
        unrealized_pnl = 0.0
        if self.current_position:
            current_price = self.dfBars['close'].iloc[-1]
            if self.current_position == OrderSide.BUY:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # OrderSide.SELL
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price

        time_elapsed = min(1.0, len(self.dfBars) / self.config.state_window)
        time_remaining = max(0.0, 1.0 - time_elapsed)

        extras = np.array([
            1.0 if self.current_position == OrderSide.BUY else -1.0 if self.current_position == OrderSide.SELL else 0.0,
            unrealized_pnl,
            time_elapsed,
            time_remaining
        ], dtype=np.float32)

        # Формируем one-hot кодирование истории действий
        hist_onehot = np.zeros(self.config.action_history_len * 4, dtype=np.float32)
        for idx, action in enumerate(self.action_history):
            hist_onehot[idx * 4 + action] = 1.0

        # Формируем окончательное состояние (аналогично TradingEnvironment._get_observation)
        normalized = normalized.reshape(self.config.num_features, self.config.input_history_len, 1)
        return np.concatenate([normalized.flatten(), extras, hist_onehot])

    def _apply_normalization(self, window: np.ndarray) -> np.ndarray:
        """Применение нормализации к окну данных"""
        if not self.normalization_stats:
            return window  # Без нормализации, если статистики не предоставлены

        normalized = np.zeros_like(window, dtype=np.float32)

        for i, channel in enumerate(self.config.data_channels):
            if channel in self.config.price_channels:
                # Нормализация цен через log returns
                prices = window[:, i]
                log_returns = np.diff(np.log(prices))
                mean = self.normalization_stats['price']['mean']
                std = self.normalization_stats['price']['std']
                normalized_returns = (log_returns - mean) / (std + 1e-8)
                normalized[1:, i] = normalized_returns
                normalized[0, i] = 0  # Первое значение не определено
            elif channel in self.config.volume_channels:
                # Нормализация объема
                volumes = window[:, i]
                mean = self.normalization_stats['volume']['mean']
                std = self.normalization_stats['volume']['std']
                normalized[:, i] = (volumes - mean) / (std + 1e-8)
            else:
                # Другие каналы (если есть)
                normalized[:, i] = window[:, i]

        return normalized

    def _get_rl_action(self, state: np.ndarray) -> int:
        """Получение действия от RL-модели"""
        with torch.no_grad():
            # Преобразуем состояние в тензор
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.agent.device)

            # Получаем Q-значения
            q_values = self.agent.policy_net(state_tensor).cpu().numpy().squeeze(0)

            # Выбираем действие с наибольшим Q-значением
            action = int(np.argmax(q_values))

            # Обновляем историю действий
            self.action_history.pop(0)
            self.action_history.append(action)

            # Логируем Q-значения
            self.log.info(
                f"Q-values: [Hold: {q_values[0]:.4f}, Long: {q_values[1]:.4f}, "
                f"Short: {q_values[2]:.4f}, Close: {q_values[3]:.4f}]",
                color=LogColor.YELLOW
            )

            return action

    def _execute_trading_decision(self, action: int, current_bar: Bar):
        """Выполнение торгового решения на основе действия модели"""
        current_price = float(current_bar.close)

        # Преобразуем действие RL-модели (0-3) в действие стратегии (0-2)
        # RL: 0=hold, 1=long, 2=short, 3=close
        # Стратегия: 0=hold, 1=buy, 2=sell

        if action == 0:  # Hold
            self.last_action = 0
            return

        elif action == 1:  # Long (BUY)
            if self.current_position != OrderSide.BUY:
                self._place_buy_order(current_price)
                self.last_action = 1

        elif action == 2:  # Short (SELL)
            if self.current_position != OrderSide.SELL:
                self._place_sell_order(current_price)
                self.last_action = 2

        elif action == 3:  # Close position
            if self.current_position:
                self._close_position(current_price)
                self.last_action = 0

    def _place_buy_order(self, price: float):
        """Размещение лимитного ордера на покупку"""
        self.log.info(f"Placing BUY order at {price:.5f}", color=LogColor.GREEN)

        # Создаем лимитный ордер
        order = self.order_factory.limit(
            instrument_id=InstrumentId.from_str(self.config.instrument_id),
            order_side=OrderSide.BUY,
            quantity=Quantity(self.config.trade_size),
            price=Price(price, precision=5),
            post_only=False
        )

        # Отправляем ордер
        self.submit_order(order)

        # Обновляем состояние позиции
        self.current_position = OrderSide.BUY
        self.entry_price = price

    def _place_sell_order(self, price: float):
        """Размещение лимитного ордера на продажу"""
        self.log.info(f"Placing SELL order at {price:.5f}", color=LogColor.RED)

        # Создаем лимитный ордер
        order = self.order_factory.limit(
            instrument_id=InstrumentId.from_str(self.config.instrument_id),
            order_side=OrderSide.SELL,
            quantity=Quantity(self.config.trade_size),
            price=Price(price, precision=5),
            post_only=False
        )

        # Отправляем ордер
        self.submit_order(order)

        # Обновляем состояние позиции
        self.current_position = OrderSide.SELL
        self.entry_price = price

    def _close_position(self, price: float):
        """Закрытие текущей позиции"""
        if not self.current_position:
            return

        side = OrderSide.SELL if self.current_position == OrderSide.BUY else OrderSide.BUY
        self.log.info(f"Closing {self.current_position.name} position at {price:.5f}", color=LogColor.MAGENTA)

        # Создаем лимитный ордер для закрытия позиции
        order = self.order_factory.limit(
            instrument_id=InstrumentId.from_str(self.config.instrument_id),
            order_side=side,
            quantity=Quantity(self.config.trade_size),
            price=Price(price, precision=5),
            post_only=False
        )

        # Отправляем ордер
        self.submit_order(order)

        # Обновляем состояние позиции
        self.current_position = None
        self.entry_price = None

    def _log_strategy_state(self, bar: Bar, action: int):
        """Логирование текущего состояния стратегии"""
        position_str = "None" if not self.current_position else self.current_position.name
        action_str = ["Hold", "Buy (Long)", "Sell (Short)", "Close"][action]

        self.log.info(
            f"Strategy State | Position: {position_str} | "
            f"Action: {action_str} | "
            f"Price: {bar.close:.5f} | "
            f"Bars in history: {len(self.dfBars)}/{self.config.state_window}",
            color=LogColor.CYAN
        )

        # Логируем последние бары для отладки
        if len(self.dfBars) >= 5:
            recent_bars = self.dfBars.tail(5).reset_index()
            recent_bars['time'] = recent_bars['time'].dt.strftime('%Y-%m-%d %H:%M')
            self.log.info(
                f"\nRecent bars:\n{tabulate(recent_bars, headers='keys', tablefmt='psql', floatfmt='.5f')}",
                color=LogColor.BLUE
            )

    def on_stop(self) -> None:
        """Действия при остановке стратегии"""
        self.log.info("Stopping RL Trading Strategy", color=LogColor.GREEN)

        # Закрываем все позиции при остановке
        if self.current_position:
            current_price = float(self.dfBars['close'].iloc[-1])
            self._close_position(current_price)

        # Освобождаем ресурсы
        if hasattr(self, 'agent'):
            del self.agent
            torch.cuda.empty_cache()

        self.log.info("RL Trading Strategy stopped", color=LogColor.GREEN)

    def on_reset(self) -> None:
        """Сброс состояния стратегии"""
        self.dfBars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.dfBars.index.name = 'time'
        self.current_position = None
        self.entry_price = None
        self.last_action = 0
        self.action_history = [0] * self.config.action_history_len
        self.log.info("Strategy state reset", color=LogColor.BLUE)
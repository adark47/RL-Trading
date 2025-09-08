# strategy.py
import os
import sys
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

# --- Импорты для RL модели ---
# Определяем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Импортируем компоненты RL
# Заменено: from config import cfg as default_cfg
import configs.alpha                # <-- Импорт нового конфига
from agent import D3QN_PER_Agent
# import configs.alpha  # <-- Убираем повторный импорт, он уже выше
from utils import select_and_arrange_channels

# Импортируем TA стратегию
# Предполагаем, что ta_strategy.py находится в поддиректории data
from data.ta_strategy import apply_strategy

# Импортируем класс модели для безопасной загрузки
from model import DuelingQNetwork
# Добавляем класс модели в список безопасных глобальных переменных для torch.load
# Это необходимо для PyTorch >= 2.3 из-за усиленной безопасности по умолчанию
import torch.serialization

torch.serialization.add_safe_globals([DuelingQNetwork])


# -----------------------------


class StrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for RLStrategy.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    primary_bar_type : BarType
        The primary bar type to subscribe to.
    trade_size : Decimal
        The fixed trade size for orders.
    model_path : str
        Path to the trained model file. Can be absolute or relative to the project root.
    """
    instrument_id: InstrumentId
    primary_bar_type: BarType
    trade_size: Decimal
    model_path: str = 'final.pth'  # Путь по умолчанию относительно корня проекта


class Strategy(Strategy):
    """
    A trading strategy that uses a trained D3QN_PER_Agent model to make decisions.
    It applies technical analysis features and uses them as input to the RL model.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize the RL strategy.

        Parameters
        ----------
        config : StrategyConfig
            The strategy configuration.
        """
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.primary_bar_type
        self.trade_size = config.trade_size


        self.model_path = config.model_path

        # Для экономии памяти
        self.max_bars_to_keep = 14400  # 10 дней по 1-минутным барам

        # Инициализация хранилища данных
        self.df_bars = pd.DataFrame()
        self.last_processed_bar_time = None

        # Компоненты RL
        self.agent = None
        self.current_state = None
        self.current_position = 0  # 0: Нет позиции, 1: Long, -1: Short

        # TA признаки, необходимые для модели
        self.required_ta_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'vol_ma', 'volume_confirmation', 'hma', 'upper_band', 'lower_band',
            'entries_1m', 'exits_1m',
            'entries_5m', 'exits_5m',
            'entries_15m', 'exits_15m',
            'entries', 'exits', 'norm_atr',
            'price_position', 'take_profit_level', 'trailing_stop_distance', 'commission'
        ]

    def on_start(self):
        """
        Actions to be performed when the strategy is started.
        Initializes the RL agent and subscribes to market data.
        """
        self.log.info("RL Strategy starting...", color=LogColor.MAGENTA)

        # Подписываемся на рыночные данные
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Subscribed to bars: {self.bar_type}", color=LogColor.BLUE)

        # Инициализируем RL агента
        try:
            self._initialize_agent()
            if self.agent is None:
                self.log.error("Failed to initialize RL agent. Strategy will not trade.")
                return

            self.log.info("RL agent successfully initialized", color=LogColor.GREEN)
        except Exception as e:
            self.log.error(f"Error initializing RL agent: {str(e)}")
            return

    def _initialize_agent(self):
        """
        Initialize the D3QN_PER_Agent with configuration from config.py.
        """
        try:
            self.log.info(f"Attempting to initialize agent with model from: {self.model_path}")

            # --- Отладка: Выводим используемые параметры конфигурации ---
            # Заменено: default_cfg на configs.alpha.cfg
            self.log.info("=== Configuration used for agent creation ===", color=LogColor.CYAN)
            self.log.info(f"state_shape: ({configs.alpha.cfg.seq.num_features}, {configs.alpha.cfg.seq.input_history_len}, 1)",
                          color=LogColor.CYAN)
            self.log.info(f"action_dim: {configs.alpha.cfg.market.num_actions}", color=LogColor.CYAN)
            self.log.info(f"cnn_maps: {configs.alpha.cfg.model.cnn_maps}", color=LogColor.CYAN)
            self.log.info(f"cnn_kernels: {configs.alpha.cfg.model.cnn_kernels}", color=LogColor.CYAN)
            self.log.info(f"cnn_strides: {configs.alpha.cfg.model.cnn_strides}", color=LogColor.CYAN)
            self.log.info(f"dense_val: {configs.alpha.cfg.model.dense_val}", color=LogColor.CYAN)
            self.log.info(f"dense_adv: {configs.alpha.cfg.model.dense_adv}", color=LogColor.CYAN)
            self.log.info(f"additional_feats: {configs.alpha.cfg.model.additional_feats}", color=LogColor.CYAN)
            self.log.info(f"device: {configs.alpha.cfg.device.device}", color=LogColor.CYAN)
            self.log.info("=============================================", color=LogColor.CYAN)
            # -----------------------------------------------------------

            # Создаем агента с параметрами из config.py
            # Заменено: default_cfg на configs.alpha.cfg
            self.agent = D3QN_PER_Agent(
                state_shape=(
                    configs.alpha.cfg.seq.num_features,
                    configs.alpha.cfg.seq.input_history_len,
                    1
                ),
                action_dim=configs.alpha.cfg.market.num_actions,
                cnn_maps=configs.alpha.cfg.model.cnn_maps,
                cnn_kernels=configs.alpha.cfg.model.cnn_kernels,
                cnn_strides=configs.alpha.cfg.model.cnn_strides,
                dense_val=configs.alpha.cfg.model.dense_val,
                dense_adv=configs.alpha.cfg.model.dense_adv,
                additional_feats=configs.alpha.cfg.model.additional_feats,
                dropout_model=configs.alpha.cfg.model.dropout_p,
                device=configs.alpha.cfg.device.device,
                gamma=configs.alpha.cfg.rl.gamma,
                learning_rate=configs.alpha.cfg.rl.learning_rate,
                batch_size=configs.alpha.cfg.rl.batch_size,
                buffer_size=configs.alpha.cfg.per.buffer_size,
                target_update_freq=configs.alpha.cfg.rl.target_update_freq,
                train_start=configs.alpha.cfg.rl.train_start,
                per_alpha=configs.alpha.cfg.per.per_alpha,
                per_beta_start=configs.alpha.cfg.per.per_beta_start,
                per_beta_frames=configs.alpha.cfg.per.per_beta_frames,
                eps_start=configs.alpha.cfg.eps.eps_start,
                eps_end=configs.alpha.cfg.eps.eps_end,
                eps_frames=configs.alpha.cfg.eps.eps_decay_frames,
                epsilon=configs.alpha.cfg.per.per_eps,
                max_gradient_norm=configs.alpha.cfg.rl.max_gradient_norm,
                backtest_cache_path=None,
            )

            # --- Отладка: Проверяем форму, которую ожидает созданная модель ---
            if self.agent and self.agent.policy_net:
                expected_input_size_value = self.agent.policy_net.value_stream[0].in_features
                expected_input_size_adv = self.agent.policy_net.advantage_stream[0].in_features
                self.log.info(f"=== Model Analysis After Creation ===", color=LogColor.CYAN)
                self.log.info(f"Policy Net Value Stream expects input size: {expected_input_size_value}",
                              color=LogColor.CYAN)
                self.log.info(f"Policy Net Advantage Stream expects input size: {expected_input_size_adv}",
                              color=LogColor.CYAN)

                # Дополнительная проверка размеров слоев
                # Проверим, совпадают ли размеры value и advantage stream
                if expected_input_size_value != expected_input_size_adv:
                    self.log.warning(
                        f"WARNING: Value stream input size ({expected_input_size_value}) != Advantage stream input size ({expected_input_size_adv})",
                        color=LogColor.YELLOW)
                self.log.info("======================================", color=LogColor.CYAN)
            # -------------------------------------------------------------------

            # Загружаем веса обученной модели
            if os.path.exists(self.model_path):
                try:
                    self.log.info("Starting model loading process...", color=LogColor.CYAN)
                    # Загружаем объект (может быть state_dict или полная модель)
                    # weights_only=False необходимо для PyTorch >= 2.3 при загрузке не-state_dict объектов
                    # Заменено: default_cfg на configs.alpha.cfg
                    loaded_obj = torch.load(self.model_path, map_location=configs.alpha.cfg.device.device, weights_only=False)
                    self.log.info("Model file loaded from disk.", color=LogColor.CYAN)

                    # Проверим тип загруженных данных
                    if isinstance(loaded_obj, dict):
                        # Это стандартный state_dict
                        self.log.info("Detected state_dict format. Attempting to load...", color=LogColor.CYAN)
                        self.agent.policy_net.load_state_dict(loaded_obj)
                        self.agent.target_net.load_state_dict(loaded_obj)
                        self.log.info(f"Model state_dict loaded successfully from {self.model_path}",
                                      color=LogColor.GREEN)
                    else:
                        # Это, вероятно, сам объект модели (DuelingQNetwork)
                        self.log.info("Detected full model object. Attempting to copy weights...", color=LogColor.CYAN)

                        # --- Отладка: Анализируем загруженную модель ---
                        if hasattr(loaded_obj, 'value_stream') and hasattr(loaded_obj, 'advantage_stream'):
                            loaded_value_in_features = loaded_obj.value_stream[0].in_features if len(
                                loaded_obj.value_stream) > 0 else 'N/A'
                            loaded_adv_in_features = loaded_obj.advantage_stream[0].in_features if len(
                                loaded_obj.advantage_stream) > 0 else 'N/A'
                            self.log.info(f"=== Loaded Model Analysis ===", color=LogColor.CYAN)
                            self.log.info(f"Loaded model Value Stream input size: {loaded_value_in_features}",
                                          color=LogColor.CYAN)
                            self.log.info(f"Loaded model Advantage Stream input size: {loaded_adv_in_features}",
                                          color=LogColor.CYAN)
                            self.log.info("==============================", color=LogColor.CYAN)

                            if loaded_value_in_features != 'N/A' and loaded_adv_in_features != 'N/A':
                                if loaded_value_in_features != expected_input_size_value:
                                    self.log.error(
                                        f"CRITICAL MISMATCH: Loaded model input size ({loaded_value_in_features}) != Current model expects ({expected_input_size_value})",
                                        color=LogColor.RED)
                                    raise ValueError("Model architecture mismatch")
                        # -----------------------------------------------

                        # Копируем state_dict из загруженной модели в наши сети
                        self.agent.policy_net.load_state_dict(loaded_obj.state_dict())
                        # Копируем веса из policy_net в target_net, как это делает стандартный load_model
                        self.agent.target_net.load_state_dict(loaded_obj.state_dict())
                        self.log.info(f"Full model object loaded and weights copied from {self.model_path}",
                                      color=LogColor.GREEN)

                    # Переводим сети в режим оценки
                    self.agent.policy_net.eval()
                    self.agent.target_net.eval()
                    self.log.info("Model networks set to evaluation mode.", color=LogColor.CYAN)

                except Exception as load_error:
                    self.log.error(f"Failed to load model from {self.model_path}: {str(load_error)}",
                                   color=LogColor.RED)
                    # Добавляем traceback для более детальной информации об ошибке
                    import traceback
                    self.log.error(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)
                    self.agent = None
            else:
                self.log.error(f"Model file not found at {self.model_path}", color=LogColor.RED)
                self.agent = None

        except Exception as e:
            self.log.error(f"Error in _initialize_agent: {str(e)}", color=LogColor.RED)
            # Добавляем traceback для более детальной информации об ошибке
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)
            self.agent = None

    def on_bar(self, bar: Bar):
        """
        Actions to be performed when the strategy receives a bar.
        Updates data, applies TA strategy, prepares state, and gets action from agent.

        Parameters
        ----------
        bar : Bar
            The received bar.
        """
        # Конвертируем временную метку бара
        bar_time = pd.Timestamp(bar.ts_init)

        # Избегаем обработки дубликатов баров
        if self.last_processed_bar_time is not None and bar_time <= self.last_processed_bar_time:
            self.log.debug(f"Skipping duplicate or old bar: {bar_time}")
            return

        self.last_processed_bar_time = bar_time

        # Добавляем новый бар в DataFrame
        new_row = pd.DataFrame([{
            'date': bar_time,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume)
        }])  # Не используем index, concat с ignore_index=True справится лучше

        # Обновляем DataFrame
        if self.df_bars.empty:
            self.df_bars = new_row
        else:
            self.df_bars = pd.concat([self.df_bars, new_row], ignore_index=True)

        # Управление памятью
        if len(self.df_bars) > self.max_bars_to_keep:
            self.df_bars = self.df_bars.tail(self.max_bars_to_keep).copy()
            self.df_bars.reset_index(drop=True, inplace=True)  # Очищаем индекс после обрезки

        self.log.info(f"DataFrame updated. Total bars: {len(self.df_bars)}", color=LogColor.CYAN)

        # Проверяем, достаточно ли данных
        # Заменено: default_cfg на configs.alpha.cfg
#        if len(self.df_bars) < configs.alpha.cfg.seq.full_seq_len:
        if len(self.df_bars) < 30:
            needed = configs.alpha.cfg.seq.full_seq_len - len(self.df_bars)
            self.log.info(
                f"Not enough data yet. Need {configs.alpha.cfg.seq.full_seq_len} bars, have {len(self.df_bars)}. Waiting for {needed} more bars.")
            return

        try:
            # Применяем стратегию технического анализа
            # Передаем путь к конфигу TA, если он находится в другом месте
            ta_config_path = '../data/ta_config_optimized.json'
            df_with_ta = apply_strategy(self.df_bars.copy(), config_path=ta_config_path)

            # Проверяем наличие всех необходимых колонок
            missing_cols = [col for col in self.required_ta_columns if col not in df_with_ta.columns]
            if missing_cols:
                self.log.error(f"Missing required TA columns: {missing_cols}")
                return

            # Подготавливаем состояние для агента
            state = self._prepare_state(df_with_ta)
            if state is None:
                return

            # Получаем действие от агента
            action = self.agent.select_action(state, training=False)
            self.log.info(f"Agent selected action: {action}", color=LogColor.YELLOW)

            # Выполняем торговую операцию на основе действия
            self._execute_trade(action, bar)

        except Exception as e:
            self.log.error(f"Error in on_bar processing: {str(e)}", color=LogColor.RED)
            # Добавляем traceback для более детальной информации об ошибке
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)

    def _prepare_state(self, df_with_ta: pd.DataFrame) -> np.ndarray:
        """
        Prepare the state vector for the RL agent.

        Parameters
        ----------
        df_with_ta : pd.DataFrame
            DataFrame with technical analysis features.

        Returns
        -------
        np.ndarray
            The prepared state vector.
        """
        try:
            # Выбираем и упорядочиваем каналы в соответствии с ожиданиями модели
            # Заменено: default_cfg на configs.alpha.cfg
            selected_data = select_and_arrange_channels(
                df_with_ta.values,
                df_with_ta.columns.tolist(),
                configs.alpha.cfg.data.data_channels
            )

            if selected_data is None:
                self.log.error("Failed to select and arrange channels")
                return None

            # Определяем границы последовательности
            # Нам нужны данные от (конец - post_signal_len - agent_history_len) до (конец - post_signal_len)
            # Где "конец" - это последний доступный индекс
            # Заменено: default_cfg на configs.alpha.cfg
            total_len = len(selected_data)
            end_idx = total_len - configs.alpha.cfg.seq.post_signal_len
            start_idx = end_idx - configs.alpha.cfg.seq.agent_history_len

            if start_idx < 0:
                self.log.error(
                    f"Not enough historical data to prepare state. Total: {total_len}, Need: {configs.alpha.cfg.seq.agent_history_len}")
                return None

            # Извлекаем окно для агента
            window = selected_data[start_idx:end_idx]

            # В реальной среде мы бы использовали предварительно рассчитанную статистику нормализации
            # Поскольку мы предполагаем, что статистика не нужна, мы пропускаем нормализацию
            # В реальной реализации здесь нужно применить нормализацию

            # Создаем дополнительные признаки
            unrealized_pnl = 0.0  # Упрощено для live-торговли
            # Предполагаем, что мы всегда на последнем шаге сессии для принятия решений
            time_elapsed = 1.0
            time_remaining = 0.0

            extras = np.array([
                float(self.current_position),  # Текущая позиция
                unrealized_pnl,  # Нереализованный PnL
                time_elapsed,  # Прошедшее время в сессии
                time_remaining,  # Оставшееся время в сессии
            ], dtype=np.float32)

            # Выравниваем (flatten) данные окна
            flat_window = window.flatten()

            # Комбинируем данные окна с дополнительными признаками
            state = np.concatenate([flat_window, extras])

            # Проверяем правильную форму
            # Заменено: default_cfg на configs.alpha.cfg
            expected_shape = (
                    configs.alpha.cfg.seq.input_history_len * configs.alpha.cfg.seq.num_features + 4
            )

            if state.shape[0] != expected_shape:
                self.log.error(f"State shape mismatch. Expected {expected_shape}, got {state.shape[0]}")
                return None

            self.log.debug(f"Prepared state with shape: {state.shape}", color=LogColor.CYAN)
            return state.astype(np.float32)

        except Exception as e:
            self.log.error(f"Error in _prepare_state: {str(e)}", color=LogColor.RED)
            # Добавляем traceback для более детальной информации об ошибке
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)
            return None

    def _execute_trade(self, action: int, bar: Bar):
        """
        Execute a trade based on the agent's action.

        Parameters
        ----------
        action : int
            The action selected by the agent.
        bar : Bar
            The current bar.
        """
        # Сопоставление действий:
        # 0: Hold, 1: Buy/Long, 2: Sell/Short, 3: Close

        if action == 1 and self.current_position == 0:  # Buy
            self.log.info("Executing BUY order", color=LogColor.GREEN)
            order = MarketOrder(
                OrderSide.BUY,
                self.instrument_id,
                self.trade_size,
                client_order_id=self.generate_client_order_id(),
                reduce_only=False
            )
            self.submit_order(order)
            self.current_position = 1

        elif action == 2 and self.current_position == 0:  # Sell
            self.log.info("Executing SELL order", color=LogColor.RED)
            order = MarketOrder(
                OrderSide.SELL,
                self.instrument_id,
                self.trade_size,
                client_order_id=self.generate_client_order_id(),
                reduce_only=False
            )
            self.submit_order(order)
            self.current_position = -1

        elif action == 3 and self.current_position != 0:  # Close
            self.log.info("Executing CLOSE order", color=LogColor.BLUE)
            side = OrderSide.SELL if self.current_position == 1 else OrderSide.BUY
            order = MarketOrder(
                side,
                self.instrument_id,
                self.trade_size,
                client_order_id=self.generate_client_order_id(),
                reduce_only=True  # Обеспечивает закрытие/сокращение позиции
            )
            self.submit_order(order)
            self.current_position = 0

        elif action == 0:
            self.log.info("Agent selected HOLD action", color=LogColor.CYAN)
        else:
            self.log.info(f"No action taken. Current position: {self.current_position}, Action: {action}")

    def on_order_filled(self, order_filled):
        """
        Actions to be performed when an order is filled.

        Parameters
        ----------
        order_filled : OrderFilled
            The order fill event.
        """
        self.log.info(f"Order filled: {order_filled}", color=LogColor.MAGENTA)

    def on_stop(self):
        """
        Actions to be performed when the strategy is stopped.
        """
        self.log.info("RL Strategy stopped", color=LogColor.MAGENTA)

    def on_reset(self):
        """
        Actions to be performed when the strategy is reset.
        """
        self.log.info("RL Strategy reset", color=LogColor.MAGENTA)
        self.df_bars = pd.DataFrame()
        self.last_processed_bar_time = None
        self.current_position = 0

    def on_dispose(self):
        """
        Actions to be performed when the strategy is disposed.
        """
        self.log.info("RL Strategy disposed", color=LogColor.MAGENTA)

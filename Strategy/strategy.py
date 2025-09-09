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
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce, PositionSide
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId, PositionId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.core.rust.model import OrderSide # Убедиться, что импорты правильные

# --- Импорты для RL модели ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем компоненты RL
import configs.alpha
from agent import D3QN_PER_Agent
from utils import select_and_arrange_channels

# Импортируем TA стратегию
from data.ta_strategy import apply_strategy

# Импортируем класс модели для безопасной загрузки
from model import DuelingQNetwork
import torch.serialization

torch.serialization.add_safe_globals([DuelingQNetwork])

# --- Оптимизация: Импорт collections для deque ---
import collections
# -------------------------------------------------

class RLStrategyConfig(StrategyConfig, frozen=True):

    instrument_id: InstrumentId
    primary_bar_type: BarType
    trade_size: Decimal
    model_path: str = 'final.pth'


class RLStrategy(Strategy):

    def __init__(self, config: RLStrategyConfig): # Уточнение типа
        """
        Initialize the RL strategy.

        Parameters
        ----------
        config : RLStrategyConfig
            The strategy configuration.
        """
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.primary_bar_type
        self.trade_size = config.trade_size

        self.model_path = config.model_path

        # --- Оптимизация: Используем deque вместо DataFrame для буфера баров ---
        # Для экономии памяти и повышения эффективности добавления/удаления
        # Определяем максимальный размер буфера (немного больше, чем нужно для TA)
        buffer_size_for_ta = max(150, configs.alpha.cfg.seq.full_seq_len + 10) # Пример, адаптируйте при необходимости
        self.bars_buffer = collections.deque(maxlen=buffer_size_for_ta)
        # -----------------------------------------------------------------------
        self.last_processed_bar_time = None

        self.agent = None
        self.current_state = None
        self.current_position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0 # Цена входа для расчета unrealized pnl

        self.order_id_counter = 0

        # Используем значение напрямую из конфига
        self.action_history_len = configs.alpha.cfg.seq.action_history_len
        # Инициализируем с None
        self.history_actions = [None] * self.action_history_len if self.action_history_len > 0 else []


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

        self.log.info("RL Strategy starting...", color=LogColor.MAGENTA)

        self.subscribe_bars(self.bar_type)
        self.log.info(f"Subscribed to bars: {self.bar_type}", color=LogColor.BLUE)

        try:
            self._initialize_agent()
            if self.agent is None:
                self.log.error("Failed to initialize RL agent. Strategy will not trade.")
                return

            self.log.info("RL agent successfully initialized", color=LogColor.GREEN)
        except Exception as e:
            self.log.error(f"Error initializing RL agent: {str(e)}")
            # Не возвращаем return в on_start, просто продолжаем (стратегия не активна)

    def _initialize_agent(self):
        """
        Initialize the D3QN_PER_Agent with configuration from config.py.
        """
        try:
            self.log.info(f"Attempting to initialize agent with model from: {self.model_path}")

            # --- Оптимизация: Уменьшено количество подробного логирования ---
            # Оставляем ключевые параметры для подтверждения конфигурации
            self.log.debug("=== Configuration used for agent creation ===", color=LogColor.CYAN)
            self.log.debug(f"state_shape: ({configs.alpha.cfg.seq.num_features}, {configs.alpha.cfg.seq.input_history_len}, 1)", color=LogColor.CYAN)
            self.log.debug(f"action_dim: {configs.alpha.cfg.market.num_actions}", color=LogColor.CYAN)
            self.log.debug("=============================================", color=LogColor.CYAN)
            # ---------------------------------------------------------------

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
                # Используем значение напрямую из конфига
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

            if os.path.exists(self.model_path):
                try:
                    self.log.info("Starting model loading process...", color=LogColor.CYAN)
                    # weights_only=False требуется, если сохранялись не state_dict
                    loaded_obj = torch.load(self.model_path, map_location=configs.alpha.cfg.device.device, weights_only=False)
                    self.log.info("Model file loaded from disk.", color=LogColor.CYAN)

                    if isinstance(loaded_obj, dict):
                        self.log.debug("Detected state_dict format. Attempting to load...", color=LogColor.CYAN)
                        self.agent.policy_net.load_state_dict(loaded_obj)
                        # Обычно для DQN target_net тоже загружается из того же state_dict
                        self.agent.target_net.load_state_dict(loaded_obj)
                        self.log.info(f"Model state_dict loaded successfully from {self.model_path}", color=LogColor.GREEN)
                    else:
                        # Предполагаем, что это сохраненный объект модели (менее вероятно для весов)
                        self.log.debug("Detected full model object. Attempting to copy weights...", color=LogColor.CYAN)
                        self.agent.policy_net.load_state_dict(loaded_obj.state_dict())
                        self.agent.target_net.load_state_dict(loaded_obj.state_dict())
                        self.log.info(f"Full model object loaded and weights copied from {self.model_path}", color=LogColor.GREEN)

                    self.agent.policy_net.eval()
                    self.agent.target_net.eval()
                    self.log.info("Model networks set to evaluation mode.", color=LogColor.CYAN)

                except Exception as load_error:
                    self.log.error(f"Failed to load model from {self.model_path}: {str(load_error)}", color=LogColor.RED)
                    import traceback
                    self.log.debug(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)
                    self.agent = None
            else:
                self.log.error(f"Model file not found at {self.model_path}", color=LogColor.RED)
                self.agent = None

        except Exception as e:
            self.log.error(f"Error in _initialize_agent: {str(e)}", color=LogColor.RED)
            import traceback
            self.log.debug(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)
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
        # Проверка дубликатов может быть важна
        bar_time = pd.Timestamp(bar.ts_init) # ts_init обычно более точный

        if self.last_processed_bar_time is not None and bar_time <= self.last_processed_bar_time:
            self.log.debug(f"Skipping duplicate or old bar: {bar_time}")
            return

        self.last_processed_bar_time = bar_time

        # --- Оптимизация: Добавление бара в deque ---
        new_bar_dict = {
            'date': bar_time,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume)
        }
        self.bars_buffer.append(new_bar_dict) # deque автоматически управляет maxlen
        # ------------------------------------------

        self.log.debug(f"Buffer updated. Total bars: {len(self.bars_buffer)}", color=LogColor.CYAN)

        # --- Оптимизация: Проверка достаточности данных ---
        # Используем длину буфера вместо DataFrame
        if len(self.bars_buffer) < configs.alpha.cfg.seq.full_seq_len:
             needed = configs.alpha.cfg.seq.full_seq_len - len(self.bars_buffer)
             self.log.info(
                 f"Not enough data yet. Need {configs.alpha.cfg.seq.full_seq_len} bars, have {len(self.bars_buffer)}. Waiting for {needed} more bars.")
             return
        # -----------------------------------------------

        try:
            # --- Оптимизация: Создание DataFrame из deque только при необходимости ---
            # Создаем DataFrame из буфера перед применением TA стратегии
            df_bars_for_ta = pd.DataFrame(list(self.bars_buffer))   # list(deque) для создания DataFrame
            # ------------------------------------------------------------------------

            # Применяем стратегию технического анализа
            ta_config_path = '../data/ta_config_optimized.json'
            # --- Потенциальная оптимизация: Проверить, модифицирует ли apply_strategy df_bars_for_ta ---
            # Если нет, можно рассмотреть передачу копии или работу с numpy массивами внутри apply_strategy
            df_with_ta = apply_strategy(df_bars_for_ta, config_path=ta_config_path)
            # ---------------------------------------------------------------------------------------

            missing_cols = [col for col in self.required_ta_columns if col not in df_with_ta.columns]
            if missing_cols:
                self.log.error(f"Missing required TA columns: {missing_cols}")
                return # Прекращаем обработку, если данные неполные

            state = self._prepare_state(df_with_ta)
            if state is None:
                return # Прекращаем, если состояние не может быть подготовлено

            # Агент выбирает действие в режиме инференса (не обучения)
            action = self.agent.select_action(state, training=False)
            self.log.info(f"Agent selected action: {action}", color=LogColor.YELLOW)

            # Обновляем историю действий
            if self.action_history_len > 0:
                # Добавляем новое действие в конец
                self.history_actions.append(action)
                # Если список переполнен, удаляем самое старое (первое)
                if len(self.history_actions) > self.action_history_len:
                     self.history_actions.pop(0) # Удаляем первый элемент

            # Выполняем торговое действие
            self._execute_trade(action, bar)

        except Exception as e:
            self.log.error(f"Error in on_bar processing: {str(e)}", color=LogColor.RED)
            import traceback
            self.log.debug(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)

    def _prepare_state(self, df_with_ta: pd.DataFrame) -> np.ndarray:
        """
        Подготавливает вектор состояния для агента на основе данных с ТА.
        """
        try:
            # Выбираем и упорядочиваем каналы данных согласно конфигурации
            selected_data = select_and_arrange_channels(
                df_with_ta.values,
                df_with_ta.columns.tolist(),
                configs.alpha.cfg.data.data_channels
            )

            if selected_data is None:
                self.log.error("Failed to select and arrange channels")
                return None

            total_len = len(selected_data)
            # Определяем окно для состояния агента
            # post_signal_len - это часть данных после сигнала, которую мы не используем для текущего состояния
            end_idx = total_len - configs.alpha.cfg.seq.post_signal_len
            # input_history_len - это длина исторических данных, которую видит агент
            start_idx = end_idx - configs.alpha.cfg.seq.input_history_len

            if start_idx < 0:
                self.log.error(
                    f"Not enough historical data to prepare state. Total: {total_len}, Need: {configs.alpha.cfg.seq.input_history_len}")
                return None

            # Извлекаем историческое окно данных
            window = selected_data[start_idx:end_idx]

            # --- Расчет дополнительных признаков ---
            # Получаем последнюю цену закрытия для расчета unrealized pnl
            # Предполагаем, что 'close' это четвертый канал (индекс 3) в data_channels
            # или используем последний элемент окна последней строки для 'close'
            # Более надежно получить индекс 'close' из data_channels
            try:
                close_col_index = configs.alpha.cfg.data.data_channels.index('close')
                # Индекс последней строки в окне
                last_row_window_idx = configs.alpha.cfg.seq.input_history_len - 1
                current_close_price = window[last_row_window_idx, close_col_index]
            except ValueError:
                self.log.error("Column 'close' not found in data_channels for state preparation.")
                return None
            except IndexError:
                self.log.error("Error accessing current close price from window.")
                return None


            # Рассчитываем unrealized pnl
            unrealized_pnl = 0.0
            if self.current_position != 0 and self.entry_price > 0:
                # Предполагаем, что позиция измеряется в контрактах/валюте, а не в стоимости
                # PnL = (Текущая цена - Цена входа) * Направление позиции * Размер позиции
                # Для упрощения, нормализуем относительно цены входа (как в env)
                 delta = (current_close_price - self.entry_price) * self.current_position
                 unrealized_pnl = delta / self.entry_price # Нормализованный PnL


            # Рассчитываем признаки времени (упрощенные, как в env)
            # Эти значения могут быть не идеальными для live-сессии, но следуют логике env
            # Предполагаем, что мы всегда "в конце" внутренней сессии агента для принятия решений
            time_elapsed = 1.0 # float(self.step_idx) / self.agent_session_len -> упрощение
            time_remaining = 0.0 # float(self.agent_session_len - self.step_idx) / self.agent_session_len -> упрощение

            # Собираем базовые экстра признаки
            extras_list = [
                float(self.current_position), # Текущая позиция агента
                unrealized_pnl,              # Нереализованный PnL
                time_elapsed,                # Прошедшее время (упрощено)
                time_remaining,              # Оставшееся время (упрощено)
            ]

            # Добавляем историю действий в one-hot кодировке
            action_history_part = []
            if self.action_history_len > 0:
                # Заполняем None действиями "ничего не делать" (0)
                temp_history = [action if action is not None else 0 for action in self.history_actions]
                num_actions = configs.alpha.cfg.market.num_actions # 4

                # Кодируем каждое действие в one-hot и добавляем к списку
                for action in temp_history:
                    one_hot = [0.0] * num_actions
                    if 0 <= action < num_actions:
                        one_hot[action] = 1.0
                    action_history_part.extend(one_hot)

            # Объединяем все экстра признаки в один массив
            extras = np.array(extras_list + action_history_part, dtype=np.float32)

            # Сплющиваем историческое окно данных
            flat_window = window.flatten()
            # Конкатенируем окно и экстра признаки для формирования полного состояния
            state = np.concatenate([flat_window, extras])

            # Проверяем размерность состояния
            expected_shape = (
                    configs.alpha.cfg.seq.input_history_len * configs.alpha.cfg.seq.num_features +
                    configs.alpha.cfg.model.additional_feats # Должно включать 4 базовых + action_history
            )

            if state.shape[0] != expected_shape:
                self.log.error(f"State shape mismatch. Expected {expected_shape}, got {state.shape[0]}. "
                               f"flat_window: {flat_window.shape}, extras: {extras.shape}")
                self.log.debug(f"Configured additional_feats: {configs.alpha.cfg.model.additional_feats}")
                self.log.debug(f"Calculated extras size: {len(extras_list)} (base) + {len(action_history_part)} (history) = {len(extras_list) + len(action_history_part)}")
                return None

            self.log.debug(f"Prepared state with shape: {state.shape}", color=LogColor.CYAN)
            # Возвращаем состояние как float32 numpy array
            return state.astype(np.float32)

        except Exception as e:
            self.log.error(f"Error in _prepare_state: {str(e)}", color=LogColor.RED)
            import traceback
            self.log.debug(f"Traceback: {traceback.format_exc()}", color=LogColor.RED)
            return None

    def _execute_trade(self, action: int, bar: Bar):
        """
        Выполняет торговое действие, вызывая соответствующие методы buy/sell.
        """
        self.log.info(f"Attempting to execute trade action: {action}", color=LogColor.BLUE)

        # Получаем инструмент для проверки деталей
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found in cache.")
            return

        # --- Логика выполнения действий ---
        # 0: Hold, 1: Buy/Go Long, 2: Sell/Go Short, 3: Close Position

        # --- Действие: Удерживать (Hold) ---
        if action == 0:
            self.log.info("Action: Hold. No trade executed.", color=LogColor.CYAN)
            return # Ничего не делаем

        # --- Действие: Закрыть позицию (Close) ---
        elif action == 3:
            if self.current_position == 0:
                self.log.info("Action: Close Position, but no position is open. No trade executed.", color=LogColor.CYAN)
                return # Нечего закрывать
            elif self.current_position == 1:
                self.log.info("Action: Close Long Position -> Selling.", color=LogColor.CYAN)
                self.sell() # Закрываем лонг продажей
                return
            elif self.current_position == -1:
                self.log.info("Action: Close Short Position -> Buying.", color=LogColor.CYAN)
                self.buy() # Закрываем шорт покупкой
                return

        # --- Действие: Открыть позицию (Buy/Long или Sell/Short) ---
        elif action in [1, 2]:
            # Проверяем, есть ли уже открытая позиция
            if self.current_position != 0:
                self.log.info(f"Action: Open Position ({'Buy' if action == 1 else 'Sell'}), but position is already {'Long' if self.current_position == 1 else 'Short'}. No trade executed.", color=LogColor.CYAN)
                return # Не открываем новую позицию, если уже есть

            # Открываем позицию
            if action == 1:
                self.log.info("Action: Open Long Position -> Buying.", color=LogColor.CYAN)
                self.buy()
            elif action == 2:
                self.log.info("Action: Open Short Position -> Selling.", color=LogColor.CYAN)
                self.sell()
            return

        # --- Неизвестное действие ---
        else:
            self.log.warning(f"Unknown action received: {action}. No trade executed.", color=LogColor.RED)
            return

    def buy(self) -> None:
        """
        Отправляет рыночный ордер на покупку.
        """
        if not self.agent:
             self.log.warning("Agent not initialized. Cannot place buy order.")
             return

        # Создаем уникальный ID для ордера
        self.order_id_counter += 1
        client_order_id = ClientOrderId(f"RL_BUY_{self.order_id_counter}")

        # Создаем рыночный ордер на покупку
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str(str(self.trade_size)), # Конвертируем Decimal в Quantity
            client_order_id=client_order_id,
            # time_in_force по умолчанию GTC для рыночных ордеров, можно опустить
        )

        # Отправляем ордер
        self.submit_order(order)
        self.log.info(f"Submitted BUY order: {order}", color=LogColor.GREEN)


    def sell(self) -> None:
        """
        Отправляет рыночный ордер на продажу.
        """
        if not self.agent:
             self.log.warning("Agent not initialized. Cannot place sell order.")
             return

        # Создаем уникальный ID для ордера
        self.order_id_counter += 1
        client_order_id = ClientOrderId(f"RL_SELL_{self.order_id_counter}")

        # Создаем рыночный ордер на продажу
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_str(str(self.trade_size)), # Конвертируем Decimal в Quantity
            client_order_id=client_order_id,
            # time_in_force по умолчанию GTC для рыночных ордеров, можно опустить
        )

        # Отправляем ордер
        self.submit_order(order)
        self.log.info(f"Submitted SELL order: {order}", color=LogColor.YELLOW) # Используем другой цвет для различия


    def on_order_filled(self, order_filled):
        """
        Обновляет внутреннее состояние при исполнении ордера.
        """
        self.log.info(f"Order filled: {order_filled}", color=LogColor.MAGENTA)

        # Получаем исполненный ордер из кэша
        order = self.cache.order(order_filled.client_order_id)
        if order is None:
            self.log.warning(f"Filled order {order_filled.client_order_id} not found in cache.")
            return

        # Получаем цену исполнения
        fill_price = order_filled.last_px.as_double() # Предполагаем, что это цена
        self.log.info(f"Fill price: {fill_price}", color=LogColor.CYAN)

        # Определяем направление действия на основе стороны ордера
        if order.side == OrderSide.BUY:
            # Покупка может означать открытие лонга или закрытие шорта
            # Для простоты предположим, что мы всегда открываем/закрываем полностью
            # TODO: Более сложная логика может потребоваться для частичных закрытий
            if self.current_position == 0: # Открываем лонг
                self.current_position = 1
                self.entry_price = fill_price
                self.log.info(f"Opened LONG position at price {self.entry_price}", color=LogColor.GREEN)
            elif self.current_position == -1: # Закрываем шорт
                self.current_position = 0
                self.entry_price = 0.0
                self.log.info("Closed SHORT position.", color=LogColor.GREEN)
            # Если была попытка купить при лонге, это ошибка логики _execute_trade

        elif order.side == OrderSide.SELL:
            # Продажа может означать открытие шорта или закрытие лонга
            if self.current_position == 0: # Открываем шорт
                self.current_position = -1
                self.entry_price = fill_price
                self.log.info(f"Opened SHORT position at price {self.entry_price}", color=LogColor.YELLOW)
            elif self.current_position == 1: # Закрываем лонг
                self.current_position = 0
                self.entry_price = 0.0
                self.log.info("Closed LONG position.", color=LogColor.YELLOW)
            # Если была попытка продать при шорте, это ошибка логики _execute_trade

        else:
            self.log.warning(f"Unknown order side filled: {order.side}")


    def on_stop(self):
        """
        Вызывается при остановке стратегии.
        """
        self.log.info("RL Strategy stopped", color=LogColor.MAGENTA)

    def on_reset(self):
        """
        Сбрасывает внутреннее состояние стратегии.
        """
        self.log.info("RL Strategy reset", color=LogColor.MAGENTA)
        # --- Оптимизация: Очистка deque ---
        self.bars_buffer.clear()
        # ---------------------------------
        self.last_processed_bar_time = None
        self.current_position = 0
        self.entry_price = 0.0      # Сброс цены входа
        # Инициализируем историю действий заново
        self.history_actions = [None] * self.action_history_len if self.action_history_len > 0 else []
        self.order_id_counter = 0

    def on_dispose(self):
        """
        Вызывается при уничтожении стратегии.
        """
        self.log.info("RL Strategy disposed", color=LogColor.MAGENTA)
        # --- Оптимизация: Очистка deque при уничтожении ---
        self.bars_buffer.clear()
        # -------------------------------------------------

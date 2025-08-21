# backtest_engine.py

import datetime as dt
import logging  # Оставляем для совместимости с MLflow
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mlflow  # Добавляем импорт MLflow
from loguru import logger  # Импортируем loguru как рекомендовано

from config import MasterConfig
from config import cfg as default_cfg
from test_agent import init_agent
from trading_environment import TradingEnvironment
from utils import (
    calculate_normalization_stats,
    create_signal_groups,
    load_config,
    load_npz_dataset,
    select_and_arrange_channels,
    set_random_seed,
)


def flatten_config(config, parent_key='', sep='.'):
    """Рекурсивно преобразует вложенный конфиг в плоский словарь с точечной нотацией."""
    items = []
    # Добавляем проверку типов для безопасной обработки разных типов конфигов
    if isinstance(config, dict):
        config_items = config.items()
    else:
        try:
            config_items = vars(config).items()
        except TypeError:
            return {}

    for k, v in config_items:
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, MasterConfig)) or (hasattr(v, '__dict__') and not isinstance(v, (list, tuple))):
            # Рекурсивно обрабатываем вложенные структуры
            nested_items = flatten_config(v, new_key, sep=sep)
            if isinstance(nested_items, dict):
                items.extend(nested_items.items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def setup_logging(cfg: MasterConfig) -> None:
    """
    Настройка логгера с использованием Loguru для улучшенного логирования.
    Создает файл 'backtest_session.log' и настраивает вывод в консоль.
    """
    log_dir = cfg.paths.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "backtest_session.log")

    # Удаляем все существующие обработчики для предотвращения дублирования
    logger.remove()

    # Добавляем файловый обработчик с ротацией
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        enqueue=True,
        serialize=False
    )

    # Добавляем цветной вывод в консоль для лучшей читаемости
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        colorize=True
    )

    logger.info("[Init] Logging system initialized with Loguru")
    logger.debug(f"Log file path: {log_file}")
    logger.debug(f"Configuration: {cfg}")


class TradeSummary:
    def __init__(self):
        self.trade_records = []
        self.logger = logger.bind(context="TradeSummary")

    def log_trade(self, info: dict, balance: float):
        ticker = info.get("ticker", "UNKNOWN")
        trade_dt = info.get("trade_dt", dt.datetime.now())
        direction = info.get("direction", "UNKNOWN")
        trade_amount = info.get("trade_amount", 0.0)
        pnl = info.get("trade_realized_pnl", 0.0)
        change_pct = (pnl / trade_amount) * 100 if trade_amount else 0.0
        balance_pct = (pnl / balance) * 100 if balance else 0.0
        price_delta_pct = info.get("trade_price_delta", 0.0) * 100

        trade_result = (
            f"{trade_dt.strftime('%Y-%m-%d %H:%M')} {direction:<5} {ticker:<12} {int(trade_amount):>6}:"
            f"   {pnl:+7.2f} ({change_pct:+6.2f}%  |{balance_pct:+7.2f}%) PRICE CHANGE: {price_delta_pct:+.2f}%"
        )
        self.trade_records.append(trade_result)
        self.logger.info(f"Trade executed: {trade_result}")


class MetricsCollector:
    def __init__(self):
        self.logger = logger.bind(context="MetricsCollector")
        self.pnl_by_day: Dict[dt.date, float] = defaultdict(float)
        self.pnl_all = []
        self.changes = []
        self.drawdowns = []
        self.trade_amounts = []
        self.balance_curve: Dict[dt.datetime, Tuple[dt.datetime, float]] = {}
        self.total_commission = 0.0
        self.correct_preds = 0
        self.total_trades = 0
        self.total_longs = 0
        self.total_shorts = 0
        self.correct_longs = 0
        self.correct_shorts = 0

    def update(self, signal_dt: dt.datetime, info: dict, balance: float):
        pnl = info.get("trade_realized_pnl", 0.0)
        commission = info.get("total_commission", 0.0)
        price_change = info.get("trade_price_delta", 0.0)
        drawdown = info.get("max_drawdown", 0.0)
        amount = info.get("trade_amount", 0.0)
        direction = info.get("direction", "UNKNOWN")
        correct = info.get("correct_prediction", False)

        self.pnl_by_day[signal_dt.date()] += pnl
        self.pnl_all.append(pnl)
        self.changes.append(price_change)
        self.drawdowns.append(drawdown)
        self.trade_amounts.append(amount)
        self.total_commission += commission
        self.total_trades += 1

        if direction == "LONG":
            self.total_longs += 1
            if correct:
                self.correct_longs += 1
        elif direction == "SHORT":
            self.total_shorts += 1
            if correct:
                self.correct_shorts += 1

        if correct:
            self.correct_preds += 1

        self.balance_curve[signal_dt] = (signal_dt, balance)

        # Детальное логирование каждой обновленной метрики
        self.logger.debug(
            f"Metrics updated: pnl={pnl:.2f}, commission={commission:.2f}, "
            f"balance={balance:.2f}, direction={direction}"
        )

    def finalize(self):
        self.logger.info("Finalizing metrics collection...")

        if not self.pnl_all:
            self.logger.warning("No PNL data available for metrics calculation")
            return {}

        pnl_all = np.array(self.pnl_all)
        pnl_by_day = np.array(list(self.pnl_by_day.values()))
        changes = np.array(self.changes)

        if not self.balance_curve:
            self.logger.warning("Balance curve is empty")
            return {}

        _, balances = zip(*sorted(self.balance_curve.values()))
        total_change = balances[-1] / balances[0] if balances[0] != 0 else 1.0
        trade_days = len(pnl_by_day)

        # Вычисляем стандартное отклонение только для отрицательных значений
        std_pnl_by_day_neg = pnl_by_day[pnl_by_day < 0].std() if np.any(pnl_by_day < 0) else 0.0
        std_pnl_all_neg = pnl_all[pnl_all < 0].std() if np.any(pnl_all < 0) else 0.0

        metrics = {
            "total_commission": f"{(-self.total_commission / balances[0]) * 100:.2f}%" if balances[0] != 0 else "0.00%",
            "avg_commission": f"{-self.total_commission / self.total_trades:.2f}" if self.total_trades > 0 else "0.00",
            "max_loss": f"{pnl_all.min():.2f}" if len(pnl_all) > 0 else "0.00",
            "max_profit": f"{pnl_all.max():.2f}" if len(pnl_all) > 0 else "0.00",
            "total_trade_days": trade_days,
            "profit_days": (
                f"{int((pnl_by_day > 0).sum())} ({((pnl_by_day > 0).sum() / trade_days) * 100:.2f}%)"
                if trade_days > 0
                else "0 (0.00%)"
            ),
            "final_balance_change": f"{(total_change - 1) * 100:.2f}%",
            "exp_day_change": (
                f"{(np.power(total_change, 1 / trade_days) - 1) * 100:.2f}%" if trade_days > 0 else "0.00%"
            ),
            "max_drawdown": f"{min(self.drawdowns) * 100:.2f}%" if self.drawdowns else "0.00%",
            "sharpe": (
                f"{(pnl_by_day.mean() / (pnl_by_day.std() + 1e-9)) * np.sqrt(len(pnl_by_day)):.2f}"
                if len(pnl_by_day) > 0
                else "0.00"
            ),
            "sortino": (
                f"{(pnl_by_day.mean() / (std_pnl_by_day_neg + 1e-9)) * np.sqrt(len(pnl_by_day)):.2f}"
                if len(pnl_by_day) > 0
                else "0.00"
            ),
            "trades_sharpe": (f"{pnl_all.mean() / (pnl_all.std() + 1e-9):.2f}" if len(pnl_all) > 0 else "0.00"),
            "trades_sortino": (f"{pnl_all.mean() / (std_pnl_all_neg + 1e-9):.2f}" if len(pnl_all) > 0 else "0.00"),
            "accuracy": (f"{self.correct_preds / self.total_trades * 100:.1f}%" if self.total_trades > 0 else "0.0%"),
            "total_trades": self.total_trades,
            "total_longs": self.total_longs,
            "total_shorts": self.total_shorts,
            "longs_correct": (
                f"{self.correct_longs} (0.0%)"
                if self.total_longs == 0
                else f"{self.correct_longs} ({(self.correct_longs / self.total_longs) * 100:.1f}%)"
            ),
            "shorts_correct": (
                f"{self.correct_shorts} (0.0%)"
                if self.total_shorts == 0
                else f"{self.correct_shorts} ({(self.correct_shorts / self.total_shorts) * 100:.1f}%)"
            ),
            "correct_avg_change": (f"{np.mean(changes[changes > 0]) * 100:.2f}%" if np.any(changes > 0) else "0.00%"),
            "correct_std_change": (f"{np.std(changes[changes > 0]) * 100:.2f}%" if np.any(changes > 0) else "0.00%"),
            "incorrect_avg_change": (
                f"{np.mean(changes[changes <= 0]) * 100:.2f}%" if np.any(changes <= 0) else "0.00%"
            ),
            "incorrect_std_change": (
                f"{np.std(changes[changes <= 0]) * 100:.2f}%" if np.any(changes <= 0) else "0.00%"
            ),
            "avg_trade_amount": (f"{np.mean(self.trade_amounts):.2f}" if len(self.trade_amounts) > 0 else "0.00"),
            "trades_per_day": (f"{self.total_trades / trade_days:.2f}" if trade_days > 0 else "0.00"),
        }

        # Логируем все метрики для отладки
        self.logger.debug("Final metrics calculated:")
        for name, value in metrics.items():
            self.logger.debug(f"  {name}: {value}")

        return metrics

    def plot_balance(self, path: str):
        if not self.balance_curve:
            self.logger.warning("Cannot plot balance curve: no data available")
            return

        try:
            times, balances = zip(*sorted(self.balance_curve.values()))
            plt.figure(figsize=(12, 6))
            plt.plot(times, balances, label="Balance", color="blue")
            plt.xlabel("Time")
            plt.ylabel("Balance")
            plt.title("Balance Over Time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()
            self.logger.info(f"Balance curve saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to plot balance curve: {str(e)}")


def get_pass_advantage(action: int, confidence: float, cfg: MasterConfig) -> bool:
    # Добавляем проверку наличия необходимых параметров в конфигурации
    long_threshold = getattr(cfg.backtest, 'long_action_threshold', 0.5)
    short_threshold = getattr(cfg.backtest, 'short_action_threshold', 0.5)
    close_threshold = getattr(cfg.backtest, 'close_action_threshold', 0.5)

    long_pass = action == 1 and confidence <= long_threshold
    short_pass = action == 2 and confidence <= short_threshold
    close_pass = action == 3 and confidence <= close_threshold
    pass_adv = long_pass or short_pass or close_pass

    logger.debug(
        f"Advantage check: action={action}, confidence={confidence:.3f}, "
        f"thresholds=[{long_threshold}, {short_threshold}, {close_threshold}], "
        f"result={pass_adv}"
    )

    return pass_adv


def run_backtest(cfg: MasterConfig) -> Dict[str, Any]:
    # Настройка логгера перед всеми операциями
    setup_logging(cfg)
    logger.info("Starting backtesting process")

    # Проверяем доступность MLflow перед использованием
    try:
        # Настройка подключения к MLflow серверу
        mlflow.set_tracking_uri("http://192.168.88.6:5500")
        mlflow.set_experiment("Backtesting")
        mlflow_available = True
        logger.info("MLflow tracking initialized successfully")
    except Exception as e:
        mlflow_available = False
        logger.warning(f"MLflow initialization failed: {str(e)}. Continuing without MLflow tracking.")

    # Начало нового MLflow run
    run_id = None
    if mlflow_available:
        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                # Логирование параметров конфигурации
                flat_config = flatten_config(cfg)
                for key, value in flat_config.items():
                    try:
                        mlflow.log_param(key, str(value))
                    except Exception as e:
                        logger.debug(f"Could not log parameter {key} to MLflow: {str(e)}")

                # Логируем конфигурацию как артефакт
                cfg_path = os.path.join(cfg.paths.log_dir, "backtest_config.yaml")
                with open(cfg_path, 'w') as f:
                    f.write(str(cfg))
                mlflow.log_artifact(cfg_path, "configs")

                logger.info(f"[MLflow] Started run with ID: {run_id} at {mlflow.get_tracking_uri()}")
        except Exception as e:
            mlflow_available = False
            logger.error(f"MLflow run failed to start: {str(e)}. Continuing without MLflow tracking.")

    # Основной код бэктеста
    set_random_seed(cfg.random_seed)
    logger.info(f"Random seed set to {cfg.random_seed}")

    try:
        # Загрузка данных для бэктеста
        logger.info("Loading backtest dataset...")
        backtest_raw = load_npz_dataset(
            file_path=cfg.paths.backtest_data_path,
            name_dataset="Backtest",
            plot_dir=cfg.paths.plot_dir,
            debug_max_size=cfg.debug.debug_max_size_data,
            plot_examples=cfg.data.plot_examples,
            plot_channel_idx=cfg.data.plot_channel_idx,
            pre_signal_len=cfg.seq.pre_signal_len,
        )
        logger.success("Backtest dataset loaded successfully")

        # Группировка данных
        logger.info("Creating signal groups...")
        grouped_backtest_data = create_signal_groups(backtest_raw)
        logger.info(f"Created {len(grouped_backtest_data)} signal groups")

        # Загрузка тренировочных данных
        logger.info("Loading training dataset for normalization stats...")
        train_raw = load_npz_dataset(
            file_path=cfg.paths.train_data_path,
            name_dataset="Train",
            plot_dir=cfg.paths.plot_dir,
            debug_max_size=cfg.debug.debug_max_size_data,
            plot_examples=0,
            plot_channel_idx=None,
            pre_signal_len=cfg.seq.pre_signal_len,
        )

        # Подготовка последовательностей
        train_seqs = []
        for _, arr in train_raw:
            sel = select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels)
            if sel is not None:
                train_seqs.append(sel)

        # Расчет статистики
        logger.info("Calculating normalization statistics...")
        stats = calculate_normalization_stats(
            train_seqs,
            cfg.data.data_channels,
            cfg.data.price_channels,
            cfg.data.volume_channels,
            cfg.data.other_channels,
        )
        logger.success("Normalization statistics calculated")

        # Определение пути к модели
        model_base = cfg.paths.extra_model_dir or cfg.paths.model_dir
        logger.debug(f"Model base directory: {model_base}")

        if not os.path.exists(model_base):
            logger.error(f"Model directory does not exist: {model_base}")
            raise FileNotFoundError(f"Model directory not found: {model_base}")

        model_folders = sorted([f for f in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, f))])
        if not model_folders:
            logger.error(f"No model folders found in: {model_base}")
            raise FileNotFoundError(f"No model folders found in: {model_base}")

        model_folder = os.path.join(model_base, model_folders[-1])
        logger.info(f"Using model folder: {model_folder}")

        # Проверка наличия модели
        best_path = os.path.join(model_folder, "best.pth")
        model_path = best_path if os.path.exists(best_path) else os.path.join(model_folder, "final.pth")

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Using model: {model_path}")

        # Инициализация агента
        logger.info("Initializing trading agent...")
        agent = init_agent(model_path, cfg, cfg.paths.extra_cache_dir or cfg.paths.cache_dir)
        logger.success("Trading agent initialized successfully")

        if cfg.backtest.clear_disk_cache:
            logger.info("Clearing disk cache as per configuration...")
            agent.clear_disk_cache()
            logger.info("Disk cache cleared")

        # Инициализация сборщика метрик и лога сделок
        result = MetricsCollector()
        trade_log = TradeSummary()
        balance = cfg.market.initial_balance
        open_sessions: List[Dict] = []

        logger.info("\n[Starting backtest execution]:")
        logger.info(f"Initial balance: {balance:.2f}")

        # Получаем пороговые значения с проверкой
        long_threshold = getattr(cfg.backtest, 'long_action_threshold', 0.5)
        short_threshold = getattr(cfg.backtest, 'short_action_threshold', 0.5)
        close_threshold = getattr(cfg.backtest, 'close_action_threshold', 0.5)
        thresholds = [long_threshold, short_threshold, close_threshold]

        logger.debug(f"Action thresholds: LONG={long_threshold}, SHORT={short_threshold}, CLOSE={close_threshold}")
        logger.debug(f"Max parallel sessions: {cfg.backtest.max_parallel_sessions}")
        logger.debug(f"Position fraction: {cfg.backtest.position_fraction}")

        # Основной цикл бэктеста
        total_signals = 0
        processed_signals = 0

        for signal_dt, signals in grouped_backtest_data.items():
            total_signals += len(signals)

            # Обновляем список открытых сессий
            current_time = signal_dt
            open_sessions = [open_s for open_s in open_sessions if open_s["end_time"] > current_time]

            # Вычисляем свободные слоты
            free_slots = cfg.backtest.max_parallel_sessions - len(open_sessions)
            logger.debug(f"Time: {signal_dt}, Open sessions: {len(open_sessions)}, Free slots: {free_slots}")

            if free_slots <= 0:
                logger.info("No free slots available, skipping signals")
                continue

            # Выбираем сигналы для обработки
            selected_signals = signals[:free_slots]
            processed_signals += len(selected_signals)

            logger.info(
                f"Processing {len(selected_signals)}/{len(signals)} signals @ {signal_dt.strftime('%Y-%m-%d %H:%M')} "
                f"for tickers: {', '.join(t for t, _ in selected_signals)}"
            )

            for ticker_name, session in selected_signals:
                position_size = balance * cfg.backtest.position_fraction
                logger.debug(f"Processing signal for {ticker_name}: position size={position_size:.2f}")

                try:
                    # Создаем торговое окружение
                    env = TradingEnvironment(
                        sequences=[session],
                        stats=stats,
                        render_mode=cfg.render_mode,
                        full_seq_len=cfg.seq.full_seq_len,
                        num_features=cfg.seq.num_features,
                        num_actions=cfg.market.num_actions,
                        flat_state_size=cfg.seq.flat_state_size,
                        initial_balance=position_size,
                        pre_signal_len=cfg.seq.pre_signal_len,
                        data_channels=cfg.data.data_channels,
                        slippage=cfg.market.slippage,
                        transaction_fee=cfg.market.transaction_fee,
                        agent_session_len=cfg.seq.agent_session_len,
                        agent_history_len=cfg.seq.agent_history_len,
                        input_history_len=cfg.seq.input_history_len,
                        price_channels=cfg.data.price_channels,
                        volume_channels=cfg.data.volume_channels,
                        other_channels=cfg.data.other_channels,
                        action_history_len=cfg.seq.action_history_len,
                        inaction_penalty_ratio=cfg.market.inaction_penalty_ratio,
                        backtest_mode=cfg.backtest_mode,
                        use_risk_management=cfg.backtest.use_risk_management,
                    )

                    obs, _ = env.reset()
                    logger.debug(f"Environment reset for {ticker_name}")

                    # Выполняем шаги в окружении
                    for step in range(cfg.seq.agent_session_len):
                        cache_key = (ticker_name, signal_dt + dt.timedelta(minutes=step))

                        # Стратегия выбора действий
                        if cfg.backtest.selection_strategy == "advantage_based_filter":
                            q_vals = agent.select_action(
                                state=obs,
                                training=False,
                                return_qvals=cfg.backtest.return_qvals,
                                use_cache=cfg.backtest.use_cache,
                                cache_key=cache_key,
                            )
                            adv = q_vals - q_vals[0]
                            action = int(np.argmax(adv))
                            confidence = adv[action]

                            pass_adv = get_pass_advantage(action, confidence, cfg)
                            if pass_adv:
                                logger.info(
                                    f"REJECTED {['LONG', 'SHORT', 'CLOSE'][action - 1]} for {ticker_name}, "
                                    f"confidence={confidence:.3f} < threshold={thresholds[action - 1]}"
                                )
                                action = 0
                        # MC-Dropout (Monte Carlo Dropout)
                        elif cfg.backtest.selection_strategy == "ensemble_q_filter":
                            q_mean, q_std = agent.predict_ensemble(
                                state=obs,
                                training=False,
                                use_cache=cfg.backtest.use_cache,
                                cache_key=cache_key,
                                n_samples=cfg.backtest.ensemble_n_samples,
                            )
                            advantage = q_mean - q_mean[0]
                            action = int(np.argmax(advantage))
                            confidence = advantage[action]
                            uncertainty = q_std[action]

                            pass_adv = get_pass_advantage(action, confidence, cfg)
                            pass_uncertainty = uncertainty >= cfg.backtest.ensemble_max_sigma
                            if pass_adv and pass_uncertainty:
                                logger.info(
                                    f"REJECTED {['LONG', 'SHORT', 'CLOSE'][action - 1]} for {ticker_name}, "
                                    f"confidence={confidence:.3f} < threshold={thresholds[action - 1]}, "
                                    f"uncertainty={uncertainty:.3f} > max_sigma_threshold={cfg.backtest.ensemble_max_sigma}"
                                )
                                action = 0

                        else:
                            action = agent.select_action(
                                state=obs,
                                training=False,
                                return_qvals=False,
                                use_cache=cfg.backtest.use_cache,
                                cache_key=cache_key,
                            )

                        # Выполняем шаг в окружении
                        obs, _, done, _, info = env.backtest_step(
                            action=action,
                            signal_dt=signal_dt,
                            ticker=ticker_name,
                            stop_loss=cfg.backtest.stop_loss,
                            take_profit=cfg.backtest.take_profit,
                            trailing_stop=cfg.backtest.trailing_stop,
                        )

                        # Обработка закрытой позиции
                        if info.get("position_closed", False):
                            info["ticker"] = ticker_name
                            trade_log.log_trade(info, balance)
                            balance += info.get("trade_realized_pnl", 0.0)
                            result.update(signal_dt + dt.timedelta(minutes=cfg.seq.agent_session_len), info, balance)
                            logger.success(
                                f"Trade closed for {ticker_name}: "
                                f"PnL={info.get('trade_realized_pnl', 0.0):.2f}, "
                                f"Balance={balance:.2f}"
                            )

                        if done:
                            break

                    # Добавляем сессию в список открытых
                    open_sessions.append({
                        "end_time": signal_dt + dt.timedelta(minutes=cfg.seq.agent_session_len),
                        "ticker": ticker_name
                    })

                except Exception as e:
                    logger.exception(f"Error processing signal for {ticker_name}: {str(e)}")
                    continue

        # Сохраняем кэш агента
        logger.info("Saving agent disk cache...")
        agent.save_disk_cache()
        logger.success("Agent disk cache saved")

        # Выводим сводку по сделкам
        logger.info("\n[Trades Summary]:")
        for trade in trade_log.trade_records:
            logger.info(trade)

        # Расчет итоговых метрик
        logger.info("\n[Final Metrics Calculation]:")
        metrics = result.finalize()

        logger.info("\n[Final Metrics Summary]:")
        for name_result, value in metrics.items():
            logger.info(f"{name_result:>23s} = {value}")

        # Логируем итоговый баланс
        logger.success(f"Backtest completed. Final balance: {balance:.2f} "
                       f"(Change: {((balance / cfg.market.initial_balance) - 1) * 100:.2f}%)")

        # Логирование результатов в MLflow
        if mlflow_available and run_id:
            try:
                logger.info("Logging results to MLflow...")

                # Логирование метрик
                for name_result, value in metrics.items():
                    # Извлекаем численное значение из строки
                    value_str = str(value).rstrip('%')
                    try:
                        # Пытаемся извлечь число из строки
                        numeric_value = None
                        for part in value_str.split():
                            try:
                                numeric_value = float(part)
                                break
                            except ValueError:
                                continue

                        if numeric_value is not None:
                            mlflow.log_metric(name_result, numeric_value)
                        else:
                            logger.debug(f"Could not extract numeric value from '{value}' for metric {name_result}")
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not convert '{value}' to numeric for {name_result}: {str(e)}")

                # Сохранение графика баланса
                if cfg.backtest.plot_backtest_balance_curve:
                    balance_curve_path = os.path.join(cfg.paths.plot_dir, "backtest_balance_curve.png")
                    result.plot_balance(balance_curve_path)
                    mlflow.log_artifact(balance_curve_path, "plots")
                    logger.info(f"[MLflow] Balance curve logged to MLflow")

                # Сохранение обученной модели
                if os.path.exists(model_path):
                    mlflow.log_artifact(model_path, "models")
                    logger.info(f"[MLflow] Model logged to MLflow")

                # Дополнительная информация
                mlflow.log_metric("total_signals", total_signals)
                mlflow.log_metric("processed_signals", processed_signals)
                mlflow.log_metric("success_rate", processed_signals / total_signals if total_signals > 0 else 0)
                mlflow.set_tag("backtest_type", "trading")
                mlflow.set_tag("run_status", "completed")
                mlflow.set_tag("mlflow.runid", run_id)

                logger.success(f"[MLflow] Results logged successfully to run ID: {run_id}")

            except Exception as e:
                logger.error(f"[MLflow] Failed to log results: {str(e)}")

        logger.info("Backtesting process completed successfully")
        return metrics

    except Exception as e:
        logger.exception("Critical error during backtesting")
        if mlflow_available and run_id:
            try:
                mlflow.set_tag("run_status", "failed")
                mlflow.log_param("error", str(e))
            except:
                pass
        raise


if __name__ == "__main__":
    try:
        # Загружаем конфигурацию
        if len(sys.argv) > 1:
            cfg = load_config(sys.argv[1])
            logger.info(f"Loaded configuration from {sys.argv[1]}")
        else:
            cfg = default_cfg
            logger.info("Using default configuration")

        # Запускаем бэктест
        run_backtest(cfg)

    except Exception as e:
        logger.exception("Application failed to start")
        sys.exit(1)
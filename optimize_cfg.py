# optimize_cfg.py
import argparse
import copy
import datetime as dt
import json
import logging
import os
import time

import optuna

from backtest_engine import run_backtest
from utils import load_config, setup_logging

parser = argparse.ArgumentParser(description="Optimise BacktestConfig parameters")
parser.add_argument("cfg_path", type=str, help="Path to experiment *.py config")
parser.add_argument("--trials", type=int, default=200, help="Total Optuna trials")
parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")

args = parser.parse_args()

base_cfg = load_config(args.cfg_path)
run_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")

session_name = "optuna_cfg_optimization_results"
opt_dir = os.path.join(base_cfg.paths.output_dir, session_name)
os.makedirs(opt_dir, exist_ok=True)

# Исправлено: заменен .dict() на .model_dump() для Pydantic V2
with open(os.path.join(opt_dir, "orig_master_cfg.json"), "w") as f:
    # json.dump(base_cfg.dict(), f, indent=2, default=str) # Старая версия
    json.dump(base_cfg.model_dump(), f, indent=2, default=str) # Новая версия

setup_logging(session_name=session_name, cfg=base_cfg)

logging.info(f"[Optuna] output dir: {opt_dir}")


def objective(trial: optuna.Trial):
    cfg = copy.deepcopy(base_cfg)
    cfg.random_seed = 17 + trial.number
    cfg.paths.extra_model_dir = base_cfg.paths.model_dir
    cfg.paths.extra_cache_dir = base_cfg.paths.cache_dir

    # SEARCH SPACE
    # b.position_fraction = trial.suggest_float("position_frac", 0.1, 1.0, step=0.05)
    # b.max_parallel_sessions = trial.suggest_int("max_sessions", 1, 8)
    cfg.backtest.long_action_threshold = trial.suggest_float("long_thr", 0.001, 0.03, log=True)
    cfg.backtest.short_action_threshold = trial.suggest_float("short_thr", 0.001, 0.03, log=True)
    cfg.backtest.close_action_threshold = trial.suggest_float("close_thr", 0.001, 0.03, log=True)
    cfg.backtest.use_risk_management = trial.suggest_categorical("use_rm", [True, False])
    if cfg.backtest.use_risk_management:
        cfg.backtest.stop_loss = trial.suggest_float("stop_loss", 0.005, 0.03)
        cfg.backtest.take_profit = trial.suggest_float("take_profit", 0.01, 0.05)
        cfg.backtest.trailing_stop = trial.suggest_float("trail", 0.001, 0.02)
    else:
        cfg.backtest.stop_loss = cfg.backtest.take_profit = cfg.backtest.trailing_stop = 0.0

    if cfg.backtest.selection_strategy == "ensemble_q_filter":
        cfg.backtest.ensemble_max_sigma = trial.suggest_float("max_sigma", 0.001, 0.015, log=True)

    cfg.paths.config_name = f"{base_cfg.paths.config_name}_trial{trial.number:05d}"
    cfg.paths.base_output_dir = opt_dir

    # for faster runs: skip plotting and example caching
    cfg.data.plot_examples = 0
    cfg.backtest.plot_backtest_balance_curve = False
    # cfg.debug.debug_max_size_data = None

    # Выполняем бэктест
    metrics = run_backtest(cfg)

    # Сохраняем все метрики как атрибуты триала
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    # TARGET METRICS - с обработкой ошибок
    try:
        # Используем .get() с дефолтными значениями для предотвращения KeyError
        total_pnl_str = metrics.get("final_balance_change", "0%")
        accuracy_str = metrics.get("accuracy", "0%")
        num_trades_str = metrics.get("total_trades", "0")

        # Конвертируем значения, обрабатываем возможные ошибки конвертации
        total_pnl = float(total_pnl_str.rstrip("%")) if isinstance(total_pnl_str, str) and total_pnl_str.endswith('%') else float(total_pnl_str)
        accuracy = float(accuracy_str.rstrip("%")) if isinstance(accuracy_str, str) and accuracy_str.endswith('%') else float(accuracy_str)
        num_trades = int(num_trades_str) if str(num_trades_str).isdigit() else int(float(num_trades_str)) # На случай, если строка "0.0"

    except (KeyError, ValueError, TypeError) as e:
        # Если метрики отсутствуют или некорректны, логируем и возвращаем нейтральные значения
        logging.warning(f"Trial {trial.number} failed to extract metrics correctly: {e}. Metrics received: {metrics}. Assigning default values.")
        total_pnl = 0.0
        accuracy = 0.0
        num_trades = 0

    # Optuna -> maximize pnl, minimize trades (multiply by -1 to minimize)
    # Цель: максимизировать PnL, минимизировать сделки (-num_trades), максимизировать точность
    return total_pnl, accuracy, -num_trades # Возвращаем кортеж значений для многокритериальной оптимизации


sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, interval_steps=2)

study = optuna.create_study(
    directions=["maximize", "maximize", "maximize"],  # pnl ↑,  accuracy ↑, −trades ↑
    sampler=sampler,
    pruner=pruner,
    study_name=f"backtest_opt_{run_stamp}",
    storage=f"sqlite:///{os.path.join(opt_dir,'optuna.db')}",
    load_if_exists=False,
)

logging.info(f"[Optuna] starting optimisation -- trials={args.trials} jobs={args.jobs}")
start_t = time.time()
study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=True)
logging.info(f"[Optuna] finished in {(time.time()-start_t)/60:.1f} min")

df = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
df.to_parquet(os.path.join(opt_dir, "trials.parquet"), index=False)

# best of Pareto front (rank 0) -> take the first one
# Фильтруем только завершенные триалы с корректными значениями
completed_trials = [t for t in study.best_trials if t.values is not None]
if completed_trials:
    best = completed_trials[0]
    best_cfg = dict(best.params)
    with open(os.path.join(opt_dir, "best_backtest_cfg.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    logging.info(f"[Optuna] best trial #{best.number}: PnL={best.values[0]:.2f}%, Accuracy={best.values[1]:.2f}%, Trades={-best.values[2]}")
    logging.info(f"[Optuna] params: {best_cfg}")
else:
    logging.warning("[Optuna] No successful trials found to determine the best configuration.")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import plot_optimization_history, plot_pareto_front

    # Проверяем, есть ли завершенные триалы перед построением графиков
    if study.trials_dataframe(attrs=("number", "values")).dropna(subset=["values"]).shape[0] > 0:
        ax1 = plot_optimization_history(study, target=lambda t: t.values[0] if t.values else None, target_name="Total PnL (%)")
        fig1 = getattr(ax1, "figure", ax1)
        fig1.savefig(os.path.join(opt_dir, "optuna_history.png"), dpi=300)
        plt.close(fig1)

        # Для Pareto фронта используем только 2 цели, если нужно
        # ax2 = plot_pareto_front(study, target_names=["PnL (%)", "Accuracy (%)", "-Trades"])
        # Или выбрать две наиболее важные цели
        # Например, PnL и -Trades
        ax2 = plot_pareto_front(study, targets=lambda t: (t.values[0], t.values[2]) if t.values else None, target_names=["PnL (%)", "-Trades"])
        fig2 = getattr(ax2, "figure", ax2)
        fig2.savefig(os.path.join(opt_dir, "pareto.png"), dpi=300)
        plt.close(fig2)
    else:
         logging.warning("No completed trials with values found, skipping plot generation.")

except Exception as e:
    logging.warning(f"Failed to draw Optuna plots: {e}")

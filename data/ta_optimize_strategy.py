# ta_optimize_strategy.py

import sys
import os

# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Импортируем конфигурацию
from config import DataPreprocessingConfig as DataPreprocessingConfig

# Создаем экземпляр конфигурации
settings = DataPreprocessingConfig()

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

import numpy as np
import json
import os
import warnings
import optuna
from datetime import datetime, timedelta
import sys
from loguru import logger
import vectorbt as vbt
from ta_strategy import apply_strategy  # Импортируем обновленную версию apply_strategy
import multiprocessing  # Для определения количества ядер
import uuid  # Для генерации уникальных имен файлов
import arcticdb as adb  # Импортируем ArcticDB
from tqdm import tqdm  # Для прогресс-бара

# Подавляем предупреждения
warnings.filterwarnings("ignore")

# Создаем директорию для логов
os.makedirs("logs", exist_ok=True)

# Настройка логгера с цветами и эмоджи
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"logs/ta_strategy_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

commission_percent = 0.0004


def load_data_from_arcticdb():
    """Загрузка данных для оптимизации из ArcticDB"""
    try:
        # === ИНИЦИАЛИЗАЦИЯ ARCTICDB ===
        storage_path = settings.ARCTIC_PATH
        logger.info(f"🔧 Инициализация ArcticDB хранилища: {storage_path}")
        ac = adb.Arctic(f"lmdb://{storage_path}")

        # Получаем библиотеку для хранения данных
        library_name = settings.LIBRARY_NAME
        if not ac.has_library(library_name):
            logger.error(f"❌ Библиотека ArcticDB '{library_name}' не найдена!")
            raise FileNotFoundError(f"Библиотека ArcticDB '{library_name}' не найдена!")

        library = ac.get_library(library_name)
        logger.info(f"🗄️ Используем библиотеку ArcticDB: {library_name}")

        # Формируем имя символа для хранения (как в get_data_to_arcticDB.py)
        symbol_name = f"{settings.TICKER}_{settings.TIMEFRAME}_{settings.MARKET_TYPE}"
        logger.info(f"🏷️ Имя символа в ArcticDB: {symbol_name}")

        # Проверяем существование символа
        if not library.has_symbol(symbol_name):
            logger.error(f"❌ Символ '{symbol_name}' не найден в библиотеке '{library_name}'!")
            raise FileNotFoundError(f"Символ '{symbol_name}' не найден в библиотеке '{library_name}'!")

        # Определение временного диапазона (добавлено)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=settings.DAYS_BACK)
        logger.info(f"📅 Запрашиваем данные за период: {start_date} - {end_date}")

        # Читаем данные
        logger.info(f"📥 Загрузка данных из ArcticDB для символа '{symbol_name}'...")
        # Загружаем все данные без фильтрации сначала
        df = library.read(symbol_name).data
        logger.debug("🔍 Данные загружены из ArcticDB")

        # Фильтруем по временному диапазону
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            logger.success(f"✅ Успешно загружено {len(df)} записей из ArcticDB за последние {settings.DAYS_BACK} дней")
        else:
            logger.error("❌ В данных отсутствует колонка 'date'")
            raise ValueError("Данные не содержат колонку 'date'")

        # Проверка, есть ли данные после фильтрации
        if len(df) == 0:
            logger.warning("⚠️ После фильтрации данные отсутствуют")
            return pd.DataFrame()

        # Убеждаемся, что 'date' является индексом или колонкой типа datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # Сортируем по дате (на всякий случай)
        df = df.sort_values('date').reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"Ошибка загрузки данных из ArcticDB: {e}")
        raise


def calculate_objective(df, params):
    """
    Расчет целевой функции для оптимизации
    """
    # Создаем директорию для временных файлов
    os.makedirs("tmp", exist_ok=True)
    # Генерируем уникальное имя для временного файла конфигурации
    temp_config_path = f"./tmp/temp_config_{uuid.uuid4().hex}.json"

    try:
        # Создаем временный конфигурационный файл с учетом всех параметров
        # Изменено: параметры для hma вместо sma
        temp_config = {
            "atr": {"timeperiod": params['atr_timeperiod']},
            # Изменено: hma вместо sma
            "hma": {"timeperiod": params['hma_timeperiod']},
            "volume_sma": {"timeperiod": params['volume_sma_timeperiod']},
            "bands": {"multiplier": params['bands_multiplier']},
            "risk_management": {
                "take_profit_percent": params['take_profit_percent'],
                "trailing_stop_percent": params['trailing_stop_percent'],
                "commission_percent": commission_percent
            }
        }

        # Сохраняем временный конфиг
        with open(temp_config_path, 'w') as f:
            json.dump(temp_config, f, indent=2)

        # Применяем стратегию с новыми параметрами
        df_with_signals = apply_strategy(df.copy(), temp_config_path)

        # Подготавливаем данные для vectorbt
        df_with_signals = df_with_signals.set_index('date')

        # Создаем портфель
        portfolio = vbt.Portfolio.from_signals(
            close=df_with_signals['close'],
            entries=df_with_signals['entries'],
            exits=df_with_signals['exits'],
            init_cash=10000,
            fees=commission_percent  # Используем фиксированную комиссию
        )

        # Получаем ключевые метрики
        stats = portfolio.stats()

        # Основная целевая функция: максимизация Win Rate
        win_rate = stats['Win Rate [%]'] if not np.isnan(stats['Win Rate [%]']) else 0
        num_trades = stats['Total Trades']
        profit_factor = stats['Profit Factor'] if not np.isnan(stats['Profit Factor']) else 0

        # Штраф за слишком малое количество сделок
        if num_trades < 5:
            return -1000

        # Комбинированная целевая функция с акцентом на Win Rate
        # Также учитываем Profit Factor и количество сделок
        objective = (win_rate / 100) * 0.6 + \
                    (profit_factor if profit_factor < 5 else 5) * 0.3 + \
                    np.sqrt(num_trades / 20) * 0.1

        return objective if not np.isnan(objective) else -1000

    except Exception as e:
        logger.warning(f"Ошибка при расчете целевой функции: {e}")
        return -1000  # Возвращаем плохое значение в случае ошибки
    finally:
        # Удаляем временный файл в блоке finally, чтобы он удалялся даже при ошибке
        try:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
                logger.debug(f"Удален временный файл конфигурации: {temp_config_path}")
        except Exception as e:
            logger.warning(f"Не удалось удалить временный файл {temp_config_path}: {e}")


def objective(trial, df):
    """
    Целевая функция для Optuna
    """
    # Определяем диапазоны параметров для оптимизации
    # Изменено: параметры для hma вместо sma
    params = {
        # Основные параметры индикаторов
        'atr_timeperiod': trial.suggest_int('atr_timeperiod', 5, 30),
        # Изменено: hma вместо sma
        'hma_timeperiod': trial.suggest_int('hma_timeperiod', 10, 100),
        'volume_sma_timeperiod': trial.suggest_int('volume_sma_timeperiod', 10, 50),
        'bands_multiplier': trial.suggest_float('bands_multiplier', 0.5, 3.0),

        # Параметры риск-менеджмента
        'take_profit_percent': trial.suggest_float('take_profit_percent', 0.1, 10.0),
        'trailing_stop_percent': trial.suggest_float('trailing_stop_percent', 0.1, 5.0)
    }

    # Ограничение: Take Profit не может быть меньше чем Commission
    # Commission в процентах (0.0004 = 0.04%)
    min_take_profit = commission_percent * 100 * 1.1  # Добавляем 10% запас для надежности
    if params['take_profit_percent'] < min_take_profit:
        params['take_profit_percent'] = min_take_profit

    # Рассчитываем значение целевой функции
    return calculate_objective(df, params)


class OptimizationProgressBar:
    """Класс для отображения прогресса оптимизации"""

    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        self.best_value = -float('inf')

    def __call__(self, study, trial):
        if study.best_value > self.best_value:
            self.best_value = study.best_value
            self.pbar.set_postfix({"Best Value": f"{self.best_value:.4f}"})
        self.pbar.update(1)

    def close(self):
        self.pbar.close()


def optimize_strategy(n_trials=100, n_jobs=1):
    """
    Оптимизация параметров стратегии

    Args:
        n_trials (int): Количество итераций оптимизации
        n_jobs (int): Количество параллельных процессов (-1 для использования всех ядер)
    """
    logger.info("🚀 Начало оптимизации параметров стратегии")
    logger.info(f"📅 Используются данные за последние {settings.DAYS_BACK} дней")

    try:
        # Загружаем данные один раз перед оптимизацией из ArcticDB
        df = load_data_from_arcticdb()  # Изменено: загрузка из ArcticDB

        if df.empty:
            logger.error("❌ Нет данных для оптимизации")
            raise ValueError("Нет данных для оптимизации")

        # Обработка n_jobs=-1
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
            logger.info(f"Используется {n_jobs} ядер процессора")
        elif n_jobs < 1:
            logger.warning(f"Недопустимое значение n_jobs={n_jobs}. Установлено значение по умолчанию n_jobs=1")
            n_jobs = 1

        # Создаем исследование Optuna
        study = optuna.create_study(
            direction='maximize',  # Максимизируем целевую функцию
            sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
        )

        # Создаем прогресс-бар
        progress_bar = OptimizationProgressBar(n_trials)

        # Определяем обертку для передачи данных
        def objective_wrapper(trial):
            return objective(trial, df)

        # Запускаем оптимизацию с параллельными вычислениями и прогресс-баром
        study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[progress_bar]
        )

        # Закрываем прогресс-бар
        progress_bar.close()

        # Получаем лучшие параметры
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"🎉 Оптимизация завершена!")
        logger.info(f"Лучшее значение целевой функции: {best_value:.4f}")
        logger.info(f"Лучшие параметры: {best_params}")

        # Сохраняем лучшие параметры в файл
        # Изменено: сохраняем hma вместо sma
        best_config = {
            "atr": {"timeperiod": best_params['atr_timeperiod']},
            # Изменено: hma вместо sma
            "hma": {"timeperiod": best_params['hma_timeperiod']},
            "volume_sma": {"timeperiod": best_params['volume_sma_timeperiod']},
            "bands": {"multiplier": best_params['bands_multiplier']},
            "risk_management": {
                "take_profit_percent": best_params['take_profit_percent'],
                "trailing_stop_percent": best_params['trailing_stop_percent'],
                "commission_percent": commission_percent
            }
        }

        with open("ta_config_optimized.json", 'w') as f:
            json.dump(best_config, f, indent=2)

        logger.info("✅ Лучшие параметры сохранены в ta_config_optimized.json")

        # Выводим дополнительную информацию
        logger.info("📊 Детали оптимизации:")
        logger.info(f"  ATR period: {best_params['atr_timeperiod']}")
        # Изменено: выводим hma вместо sma
        logger.info(f"  HMA period: {best_params['hma_timeperiod']}")
        logger.info(f"  Volume SMA period: {best_params['volume_sma_timeperiod']}")
        logger.info(f"  Bands multiplier: {best_params['bands_multiplier']:.2f}")
        logger.info(f"  Take Profit: {best_params['take_profit_percent']:.2f}%")
        logger.info(f"  Trailing Stop: {best_params['trailing_stop_percent']:.2f}%")
        logger.info(f"  Commission: {commission_percent * 100:.3f}%")

        return best_params, best_value

    except Exception as e:
        logger.error(f"❌ Ошибка при оптимизации: {e}")
        raise


if __name__ == "__main__":
    # Запуск оптимизации с параллельными вычислениями
    # Используйте n_jobs=-1 для автоматического определения количества ядер
    best_params, best_value = optimize_strategy(n_trials=settings.n_trials, n_jobs=settings.n_jobs)
# ta_stats.py

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
from tabulate import tabulate  # Импортируем tabulate
from loguru import logger
from datetime import datetime, timedelta
import sys
import os
import vectorbt as vbt
import arcticdb as adb  # Импортируем ArcticDB

# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Импортируем конфигурацию
from config import DataPreprocessingConfig as DataPreprocessingConfig

# Создаем экземпляр конфигурации
settings = DataPreprocessingConfig()

# Создаем директорию для логов
os.makedirs("logs", exist_ok=True)
os.makedirs("optimization_results", exist_ok=True)

# Настройка логгера с цветами и эмоджи
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"logs/ta_strategy_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

# Импортируем стратегию из отдельного файла
from ta_strategy import apply_strategy


# -------------------------------
# Функция для загрузки данных из ArcticDB
# -------------------------------
def load_data_from_arcticdb():
    """Загрузка данных из ArcticDB с фильтрацией по DAYS_BACK"""
    try:
        # Инициализация ArcticDB
        storage_path = settings.ARCTIC_PATH
        logger.info(f"🔧 Подключение к ArcticDB хранилищу: {storage_path}")
        ac = adb.Arctic(f"lmdb://{storage_path}")

        # Получаем библиотеку
        library_name = settings.LIBRARY_NAME
        if not ac.has_library(library_name):
            logger.error(f"❌ Библиотека ArcticDB '{library_name}' не найдена")
            raise Exception(f"Библиотека ArcticDB '{library_name}' не найдена")

        library = ac.get_library(library_name)
        logger.info(f"🗄️ Используем библиотеку ArcticDB: {library_name}")

        # Формируем имя символа
        symbol_name = f"{settings.TICKER}_{settings.TIMEFRAME}_{settings.MARKET_TYPE}"
        logger.info(f"🏷️ Имя символа в ArcticDB: {symbol_name}")

        # Проверяем наличие символа
        if not library.has_symbol(symbol_name):
            logger.error(f"❌ Символ '{symbol_name}' не найден в библиотеке '{library_name}'")
            raise Exception(f"Символ '{symbol_name}' не найден в библиотеке '{library_name}'")

        # Определение временного диапазона если задан DAYS_BACK
        if hasattr(settings, 'DAYS_BACK') and settings.DAYS_BACK > 0:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=settings.DAYS_BACK)
            logger.info(f"📅 Запрашиваем данные за период: {start_date} - {end_date}")
        else:
            start_date = None
            end_date = None
            logger.info("📅 Загрузка всех доступных данных (без ограничения по времени)")

        # Читаем данные
        if start_date and end_date:
            # Загружаем данные с фильтрацией по дате
            arctic_result = library.read(symbol_name)
            df = arctic_result.data

            # Фильтруем по временному диапазону
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                logger.info(
                    f"📥 Загружено {len(df)} строк данных из ArcticDB ({symbol_name}) за последние {settings.DAYS_BACK} дней")
            else:
                logger.warning("⚠️ В данных отсутствует колонка 'date', загружаем все данные")
                df = library.read(symbol_name).data
        else:
            # Загружаем все данные
            df = library.read(symbol_name).data
            logger.info(f"📥 Загружено {len(df)} строк данных из ArcticDB ({symbol_name})")

        if not df.empty:
            logger.info(f"📊 Диапазон данных: с {df['date'].min()} по {df['date'].max()}")
        else:
            logger.warning("⚠️ Загружен пустой набор данных")

        # Сортируем по дате (на всякий случай)
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных из ArcticDB: {str(e)}")
        raise


# -------------------------------
# Функция для подсчета сигналов и вывода таблицы
# -------------------------------
def count_and_display_signals(df):
    """
    Подсчитывает количество сигналов и выводит их в виде таблицы.
    """
    logger.info("🔢 Подсчет сигналов...")
    try:
        signal_columns = ['long_entries', 'long_exits', 'short_entries', 'short_exits']

        # Проверяем наличие колонок
        missing_cols = [col for col in signal_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Следующие колонки сигналов отсутствуют в данных: {missing_cols}")
            # Создаем их с False, если отсутствуют
            for col in missing_cols:
                df[col] = False

        # Подсчитываем True значения в каждой колонке
        signal_counts = {col: df[col].sum() if col in df.columns else 0 for col in signal_columns}

        # Подготавливаем данные для таблицы
        table_data = [
            ["Сигнал", "Количество"],
            ["long_entries", signal_counts.get('long_entries', 0)],
            ["long_exits", signal_counts.get('long_exits', 0)],
            ["short_entries", signal_counts.get('short_entries', 0)],
            ["short_exits", signal_counts.get('short_exits', 0)],
        ]

        # Выводим таблицу с использованием tabulate
        table_str = tabulate(table_data, headers='firstrow', tablefmt='grid')
        logger.info(f"📊 Таблица подсчета сигналов:\n{table_str}")

        return signal_counts

    except Exception as e:
        logger.error(f"❌ Ошибка при подсчете сигналов: {str(e)}")
        raise


# -------------------------------
# Функция для генерации статистики с использованием vectorbt
# -------------------------------
def generate_strategy_stats(df):
    """
    Генерация статистики стратегии с использованием vectorbt.
    Ожидает наличие колонок: 'long_entries', 'long_exits', 'short_entries', 'short_exits'
    """
    logger.info("📊 Генерация статистики стратегии с использованием vectorbt")

    try:
        # Убедимся, что индекс - это DatetimeIndex
        if 'date' in df.columns:
            df = df.set_index('date')

        # Проверим, что индекс является DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("❌ Индекс DataFrame должен быть DatetimeIndex для vectorbt")
            raise ValueError("Индекс DataFrame должен быть DatetimeIndex для vectorbt")

        # Создание портфеля vectorbt с явным указанием freq
        # Определяем частоту данных для избежания ошибок с timedelta
        if len(df) > 1:
            freq = pd.infer_freq(df.index)
            if freq is None:
                # Если не удается определить частоту, используем минимальную разницу
                diffs = df.index.to_series().diff().dropna()
                if len(diffs) > 0:
                    freq = diffs.min()
                else:
                    freq = '1D'  # По умолчанию ежедневная частота
        else:
            freq = '1D'

        logger.info(f"📈 Частота данных определена как: {freq}")

        # --- Изменения здесь ---
        # Проверим, что необходимые колонки сигналов существуют в DataFrame
        required_signal_columns = ['long_entries', 'long_exits', 'short_entries', 'short_exits']
        missing_columns = [col for col in required_signal_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ В DataFrame отсутствуют следующие необходимые столбцы сигналов: {missing_columns}")
            raise ValueError(f"В DataFrame отсутствуют следующие необходимые столбцы сигналов: {missing_columns}")
        # --- Конец изменений ---

        # --- Изменения здесь ---
        # Создаем портфель с сигналами Long & Short
        # Используем from_signals с параметрами для long и short позиций
        portfolio = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=df['long_entries'],  # Сигналы открытия длинных позиций
            exits=df['long_exits'],  # Сигналы закрытия длинных позиций
            short_entries=df['short_entries'],  # Сигналы открытия коротких позиций
            short_exits=df['short_exits'],  # Сигналы закрытия коротких позиций
            freq=freq,  # Явно указываем частоту
            init_cash=10000,
            fees=0.0004,  # Комиссии 0.04% (0.1% было в оригинале, скорректировал как в комментарии)
            # --- Дополнительные параметры для управления капиталом ---
            # cash_sharing=True,             # Если нужно использовать общий капитал для всех символов (для мульти-ассет)
            # exclusive_orders=True,         # Если нужно предотвращать одновременные лонг и шорт позиции (опционально)
        )
        # --- Конец изменений ---

        # Генерация статистики
        stats = portfolio.stats()

        # Вывод статистики
        logger.info("📈 Статистика портфеля (Long & Short):")
        logger.info(f"\n{stats}")

        return stats, portfolio

    except Exception as e:
        logger.error(f"❌ Ошибка при генерации статистики: {str(e)}")
        raise


# -------------------------------
# Основной процесс
# -------------------------------
if __name__ == "__main__":
    try:
        logger.info("🚀 Запуск торговой стратегии")
        logger.info(f"⚙️ Используем DAYS_BACK: {getattr(settings, 'DAYS_BACK', 'Не задано')} дней")

        # Загрузка данных из ArcticDB
        df = load_data_from_arcticdb()
        if df is None or df.empty:
            raise Exception("Не удалось загрузить данные из ArcticDB")

        # Применение стратегии с оптимизированными параметрами и учетом комиссий
        logger.info("⚙️ Применение стратегии с оптимизированными параметрами и учетом комиссий...")
        df_result = apply_strategy(df)

        logger.info(f"Загружено {len(df_result)} записей с {df_result['date'].min()} по {df_result['date'].max()}")
        logger.info(f"📊 Размер данных: {df_result.shape} (строк: {df_result.shape[0]}, колонок: {df_result.shape[1]})")

        logger.info(f"📋 Список колонок в данных: {list(df_result.columns)}")

        # --- НОВАЯ СТРОКА ---
        # Подсчет и вывод таблицы сигналов
        signal_counts = count_and_display_signals(df_result)
        # --- КОНЕЦ НОВОЙ СТРОКИ ---

        # Генерация статистики с использованием vectorbt
        stats, portfolio = generate_strategy_stats(df_result)

    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {str(e)}")
        raise
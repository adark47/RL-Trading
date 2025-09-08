# ta_stats.py

import pandas as pd
import numpy as np
from tabulate import tabulate
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
# Функция для генерации статистики с использованием vectorbt
# -------------------------------
def generate_strategy_stats(df):
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

        # Проверим, что сигналы существуют в DataFrame
        if 'entries' not in df.columns or 'exits' not in df.columns:
            logger.error("❌ В DataFrame отсутствуют столбцы 'entries' или 'exits'")
            raise ValueError("В DataFrame отсутствуют столбцы 'entries' или 'exits'")

        # Создаем портфель с правильной частотой
        portfolio = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=df['entries'],
            exits=df['exits'],
            freq=freq,  # Явно указываем частоту
            init_cash=10000,
            fees=0.001  # Комиссии 0.1%
        )

        # Генерация статистики
        stats = portfolio.stats()

        # Вывод статистики
        logger.info("📈 Статистика портфеля:")
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

        # Генерация статистики с использованием vectorbt
        stats, portfolio = generate_strategy_stats(df_result)

    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {str(e)}")
        raise
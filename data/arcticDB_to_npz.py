# data/csv_to_npz.py

from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import time
import talib
import math
from pathlib import Path
import arcticdb as adb  # Импортируем ArcticDB

# Настройка логгера Loguru с эмодзи и улучшенным цветовым оформлением
log_dir = "logs"
logger.remove()  # Удаляем стандартный обработчик
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"{log_dir}/arcticdb_to_npz_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


class FinancialDataPreprocessor:
    """Класс для преобразования данных из ArcticDB в формат .npz для ML-моделей"""

    def __init__(self, ticker: str = "DOGEUSDT", window_size: int = 150,
                 timeframe: str = "1m", market_type: str = "linear",
                 arctic_path: str = "arcticdb_storage",
                 library_name: str = "bybit_market_data",
                 days_back: float = 30.0):
        self.ticker = ticker
        self.window_size = window_size
        self.timeframe = timeframe
        self.market_type = market_type
        self.arctic_path = arctic_path
        self.library_name = library_name
        self.days_back = days_back
        self.symbol_name = f"{ticker}_{timeframe}_{market_type}"

        # Базовые колонки, которые будут расширены техническими индикаторами
        self.feature_columns = ["open", "high", "low", "close", "volume"]
        self.logger = logger.bind(component="FinancialPreprocessor")

        # Инициализация ArcticDB
        self._init_arcticdb()

    def _init_arcticdb(self):
        """Инициализация подключения к ArcticDB"""
        try:
            self.logger.info(f"🔧 Инициализация ArcticDB хранилища: {self.arctic_path}")
            self.ac = adb.Arctic(f"lmdb://{self.arctic_path}")

            if not self.ac.has_library(self.library_name):
                self.logger.error(f"❌ Библиотека '{self.library_name}' не найдена в ArcticDB")
                raise ValueError(f"Библиотека '{self.library_name}' не существует")

            self.library = self.ac.get_library(self.library_name)
            self.logger.success(f"🗄️ Успешно подключено к библиотеке: {self.library_name}")
        except Exception as e:
            self.logger.exception(f"🔥 Ошибка инициализации ArcticDB: {str(e)}")
            raise


    @logger.catch
    def validate_data(self, df: pd.DataFrame) -> None:
        """Проверка целостности данных с расширенной информацией об ошибках"""
        self.logger.debug("🔍 Начало валидации данных")

        # Проверка наличия обязательных колонок
        required_columns = ["date"] + self.feature_columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"❌ Отсутствуют необходимые колонки: {missing}")
            raise ValueError(f"Отсутствуют необходимые колонки: {missing}")

        # Проверка формата даты
        try:
            df["date"] = pd.to_datetime(df["date"])
            self.logger.debug("📅 Дата успешно преобразована в datetime формат")
        except Exception as e:
            self.logger.error(f"❌ Ошибка преобразования даты: {str(e)}")
            raise ValueError("Колонка 'date' имеет некорректный формат") from e

        # Проверка на наличие пропусков (после обработки NaN)
        null_counts = df[self.feature_columns].isnull().sum()
        total_null = null_counts.sum()

        if total_null > 0:
            self.logger.warning(f"⚠️ Обнаружено {total_null} пропущенных значений")
            for col, count in null_counts.items():
                if count > 0:
                    self.logger.debug(f"🔍 Пропуски в '{col}': {count}")
        else:
            self.logger.success("✅ Нет пропущенных значений в данных")

        # Проверка временного порядка
        if not df['date'].is_monotonic_increasing:
            self.logger.warning("⚠️ Данные не отсортированы по возрастанию даты")
            df = df.sort_values('date').reset_index(drop=True)
            self.logger.info("🔄 Данные отсортированы по дате")

        self.logger.success("✅ Валидация данных завершена успешно")

    @logger.catch
    def create_windows(self, df: pd.DataFrame, dataset_type: str) -> Tuple[
        List[np.ndarray], Dict[str, Tuple[str, datetime]]]:
        """Создание временных окон из данных"""
        self.logger.info(f"🔄 Создание окон для {dataset_type} (размер окна: {self.window_size})")

        windows = []
        keys_map = {}
        total_rows = len(df)

        if total_rows < self.window_size:
            self.logger.error(
                f"❌ Недостаточно данных для создания окон (требуется {self.window_size}, доступно {total_rows})")
            return [], {}

        # Прогресс-бар для больших наборов данных
        start_time = time.time()
        progress_interval = max(1, (total_rows - self.window_size) // 10)

        for i in range(total_rows - self.window_size + 1):
            if i % progress_interval == 0:
                progress = (i + 1) / (total_rows - self.window_size + 1) * 100
                self.logger.debug(
                    f"⏳ Прогресс создания окон: {progress:.1f}% ({i + 1}/{total_rows - self.window_size + 1})")

            window_data = df.iloc[i:i + self.window_size][self.feature_columns].values
            last_date = df.iloc[i + self.window_size - 1]["date"]

            # Сохранение окна и соответствующего ключа
            windows.append(window_data)
            keys_map[str(len(windows) - 1)] = (self.ticker, last_date)

        elapsed = time.time() - start_time
        self.logger.success(f"✅ Создано {len(windows)} окон за {elapsed:.2f} секунд 🚀")
        return windows, keys_map

    @logger.catch
    def save_npz(self, windows: List[np.ndarray], keys_map: Dict, output_path: str) -> None:
        """Сохранение данных в формате .npz"""
        self.logger.info(f"💾 Сохранение {len(windows)} окон в {output_path}")

        # Подготовка структуры данных
        arrays = {str(i): window for i, window in enumerate(windows)}
        arrays["_keys_map_"] = keys_map

        # Проверка размеров
        if windows:
            sample = windows[0]
            self.logger.debug(f"📐 Формат данных: {sample.shape} (пример для окна 0)")

        # Сохранение с измерением времени
        start_time = time.time()
        np.savez_compressed(output_path, **arrays)
        elapsed = time.time() - start_time

        # Проверка размера файла
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        self.logger.success(f"✅ Файл сохранен: {output_path} ({file_size:.2f} MB, {elapsed:.2f} сек) 📦")

    @logger.catch
    def load_data_from_arcticdb(self) -> pd.DataFrame:
        """Загрузка данных из ArcticDB с возможностью указания глубины"""
        self.logger.info(f"📥 Загрузка данных из ArcticDB для {self.symbol_name}")
        self.logger.info(f"⚙️ Параметры: {self.days_back} дней, таймфрейм {self.timeframe}")

        # Определение временного диапазона
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)

        self.logger.info(f"📅 Запрашиваем данные за период: {start_date} - {end_date}")

        # Проверка существования символа
        if not self.library.has_symbol(self.symbol_name):
            self.logger.error(f"❌ Символ '{self.symbol_name}' не найден в библиотеке '{self.library_name}'")
            raise ValueError(f"Символ '{self.symbol_name}' не существует в библиотеке")

        # Загрузка данных с фильтрацией по времени
        # В ArcticDB 2.x используем кортеж (start, end) вместо DateRange
        try:
            # Загружаем все данные без фильтрации сначала
            self.logger.debug("🔍 Загружаем данные из ArcticDB...")
            data = self.library.read(self.symbol_name).data

            # Фильтруем по временному диапазону
            if 'date' in data.columns:
                data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
            else:
                self.logger.error("❌ В данных отсутствует колонка 'date'")
                raise ValueError("Данные не содержат колонку 'date'")

            self.logger.success(f"✅ Загружено {len(data)} записей за {self.days_back} дней")

            # Проверка, есть ли данные после фильтрации
            if len(data) == 0:
                self.logger.warning("⚠️ После фильтрации данные отсутствуют")
                return pd.DataFrame()

            return data

        except Exception as e:
            self.logger.exception(f"🔥 Ошибка загрузки данных из ArcticDB: {str(e)}")
            raise

    @logger.catch
    def process_dataset(self, output_files: Dict[str, str], percentages: Dict[str, float]) -> None:
        """Основной метод обработки данных из ArcticDB"""
        self.logger.info(f"🚀 Начало обработки данных из ArcticDB")
        start_time = time.time()

        try:
            # Загрузка данных из ArcticDB
            df = self.load_data_from_arcticdb()

            if df.empty:
                self.logger.error("❌ Нет данных для обработки после загрузки из ArcticDB")
                return

            # Преобразование и сортировка
            self.logger.info("🔄 Преобразование и сортировка данных")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self.logger.info(f"📅 Отсортировано {len(df)} записей с {df['date'].min()} по {df['date'].max()}")

            # Проверка временного разрешения
            time_diff = df["date"].diff().min()
            self.logger.debug(f"⏱️ Минимальный интервал данных: {time_diff}")

#            # Добавление технических индикаторов (ПОСЛЕ сортировки!)
#            df = self.add_technical_indicators(df)

            # Валидация данных
            self.validate_data(df)

            # Разделение данных
            self.logger.info("✂️ Разделение данных на наборы")
            total = len(df)
            cumulative = 0
            splits = {}

            # Проверка суммы процентов
            total_percent = sum(percentages.values())
            if abs(total_percent - 1.0) > 0.01:
                self.logger.warning(f"⚠️ Сумма процентов ({total_percent:.2f}) не равна 1.0, корректировка")
                percentages = {k: v / total_percent for k, v in percentages.items()}

            for dataset, percent in percentages.items():
                if percent <= 0:
                    continue

                segment_size = int(total * percent)
                end_idx = min(cumulative + segment_size, total)
                splits[dataset] = (cumulative, end_idx)
                cumulative = end_idx
                self.logger.info(f"🎯 {dataset}: {percent:.1%} ({segment_size} записей) [{cumulative}/{total}]")

            # Обработка каждого набора
            for dataset, (start_idx, end_idx) in splits.items():
                if dataset not in output_files:
                    continue

                self.logger.info(f"🔧 Обработка набора '{dataset}'")
                segment = df.iloc[start_idx:end_idx].copy()

                # Создание окон
                windows, keys_map = self.create_windows(segment, dataset)

                # Сохранение
                if windows:
                    self.save_npz(windows, keys_map, output_files[dataset])
                else:
                    self.logger.warning(f"⚠️ Пропуск сохранения {dataset}: нет данных для окон")

            total_time = time.time() - start_time
            self.logger.success(f"✅ Обработка завершена успешно за {total_time:.2f} секунд! 🎯")

        except Exception as e:
            self.logger.exception(f"💥 Критическая ошибка при обработке данных: {str(e)}")
            raise


# Пример использования
if __name__ == "__main__":
    logger.info("🚀 Запуск Financial Data Preprocessor для ArcticDB")

    try:
        # Параметры подключения к ArcticDB
        ARCTIC_PATH = "arcticdb_storage"
        LIBRARY_NAME = "bybit_market_data"
        TICKER = "DOGEUSDT"
        TIMEFRAME = "1m"
        MARKET_TYPE = "linear"
        DAYS_BACK = 100.0  # Глубина запроса в днях

        # Исправленные проценты
        percentages = {
            "train": 0.75,
            "val": 0.05,
            "test": 0.10,
            "backtest": 0.10
        }

        # Проверка суммы процентов
        total_percent = sum(percentages.values())
        if abs(total_percent - 1.0) > 0.01:
            logger.warning(f"⚠️ Сумма процентов ({total_percent:.2f}) не равна 1.0")

        preprocessor = FinancialDataPreprocessor(
            ticker=TICKER,
            window_size=150,
            timeframe=TIMEFRAME,
            market_type=MARKET_TYPE,
            arctic_path=ARCTIC_PATH,
            library_name=LIBRARY_NAME,
            days_back=DAYS_BACK
        )

        preprocessor.process_dataset(
            output_files={
                "train": "train_data.npz",
                "val": "val_data.npz",
                "test": "test_data.npz",
                "backtest": "backtest_data.npz"
            },
            percentages=percentages
        )

        logger.success("🎉 Все файлы успешно созданы с техническими индикаторами!")

    except Exception as e:
        logger.critical(f"💥 Завершение работы из-за критической ошибки: {str(e)}")
        sys.exit(1)
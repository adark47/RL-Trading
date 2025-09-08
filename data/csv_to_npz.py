
import sys
import os
# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Импортируем конфигурацию
from config import DataPreprocessingConfig as DataPreprocessingConfig
# Создаем экземпляр конфигурации
settings = DataPreprocessingConfig()

from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate
from typing import Dict, List, Tuple
import sys
import time
from pathlib import Path

from ta_strategy import apply_strategy

# Настройка логгера Loguru
log_dir = "logs"
logger.remove()  # Удаляем стандартный обработчик
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    f"{log_dir}/csv_to_npz_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


class FinancialDataPreprocessor:
    """Класс для преобразования CSV с финансовыми данными в формат .npz для ML-моделей"""

    def __init__(self, ticker: str = "STOCK", window_size: int = 150):
        self.ticker = ticker
        self.window_size = window_size

        # Базовые колонки, которые будут расширены техническими индикаторами
        # Это начальное значение, будет обновлено после apply_strategy
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']

        self.logger = logger.bind(component="FinancialPreprocessor")

    @logger.catch
    def validate_data(self, df: pd.DataFrame) -> None:
        """Проверка целостности данных"""
        self.logger.debug("Начало валидации данных")

        # Проверка наличия обязательных колонок (включая обновленные feature_columns)
        required_columns = ["date"] + self.feature_columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"Отсутствуют необходимые колонки: {missing}")
            raise ValueError(f"Отсутствуют необходимые колонки: {missing}")

        # Проверка формата даты
        try:
            pd.to_datetime(df["date"])
        except Exception as e:
            self.logger.error(f"Ошибка преобразования даты: {str(e)}")
            raise ValueError("Колонка 'date' имеет некорректный формат") from e

        # Проверка на наличие пропусков
        null_counts = df[self.feature_columns].isnull().sum()
        total_null = null_counts.sum()
        if total_null > 0:
            self.logger.warning(f"Обнаружено {total_null} пропущенных значений")
            for col, count in null_counts.items():
                if count > 0:
                    self.logger.debug(f"Пропуски в '{col}': {count}")
        self.logger.info(
            f"Превью данных: \n{tabulate(df.tail(10), headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False)}")

        self.logger.success("Валидация данных завершена успешно")

    @logger.catch
    def create_windows(self, df: pd.DataFrame, dataset_type: str) -> Tuple[
        List[np.ndarray], Dict[str, Tuple[str, datetime]]]:
        """Создание временных окон из данных"""
        self.logger.info(f"Создание окон для {dataset_type} (размер окна: {self.window_size})")

        windows = []
        keys_map = {}
        total_rows = len(df)

        if total_rows < self.window_size:
            self.logger.error(
                f"Недостаточно данных для создания окон (требуется {self.window_size}, доступно {total_rows})")
            return [], {}

        # Прогресс-бар для больших наборов данных
        start_time = time.time()
        progress_interval = max(1, (total_rows - self.window_size) // 10)

        for i in range(total_rows - self.window_size + 1):
            if i % progress_interval == 0:
                progress = (i + 1) / (total_rows - self.window_size + 1) * 100
                self.logger.debug(
                    f"Прогресс создания окон: {progress:.1f}% ({i + 1}/{total_rows - self.window_size + 1})")

            # Используем обновленный self.feature_columns
            window_data = df.iloc[i:i + self.window_size][self.feature_columns].values
            last_date = df.iloc[i + self.window_size - 1]["date"]

            # Сохранение окна и соответствующего ключа
            windows.append(window_data)
            keys_map[str(len(windows) - 1)] = (self.ticker, last_date)

        elapsed = time.time() - start_time
        self.logger.success(f"Создано {len(windows)} окон за {elapsed:.2f} секунд")
        return windows, keys_map

    @logger.catch
    def save_npz(self, windows: List[np.ndarray], keys_map: Dict, output_path: str) -> None:
        """Сохранение данных в формате .npz"""
        self.logger.info(f"Сохранение {len(windows)} окон в {output_path}")

        # Подготовка структуры данных
        arrays = {str(i): window for i, window in enumerate(windows)}
        arrays["_keys_map_"] = keys_map

        # Проверка размеров
        if windows:
            sample = windows[0]
            self.logger.debug(f"Формат данных: {sample.shape} (пример для окна 0)")

        # Сохранение с измерением времени
        start_time = time.time()
        np.savez_compressed(output_path, **arrays)
        elapsed = time.time() - start_time

        # Проверка размера файла
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        self.logger.success(f"Файл сохранен: {output_path} ({file_size:.2f} MB, {elapsed:.2f} сек)")

    @logger.catch
    def process_dataset(self, csv_path: str, output_files: Dict[str, str], percentages: Dict[str, float]) -> None:
        """Основной метод обработки данных"""
        self.logger.info(f"Начало обработки данных из {csv_path}")
        start_time = time.time()

        try:
            # Загрузка данных
            self.logger.info("Загрузка данных из CSV")
            df = pd.read_csv(csv_path)

            logger.info("⚙️ Применение стратегии с оптимизированными параметрами и учетом комиссий...")
            df_before_strategy = df.copy()  # Сохраняем копию для сравнения
            df = apply_strategy(df)  # Предполагаем, что apply_strategy модифицирует df, добавляя новые колонки

            # Определяем, какие колонки были добавлены apply_strategy (кроме базовых и даты)
            base_and_date_cols = set(["open", "high", "low", "close", "volume", "date"])
            all_cols = set(df.columns)
            added_feature_cols = sorted(list(all_cols - base_and_date_cols))  # Сортируем для консистентности

            # Обновляем self.feature_columns: базовые + новые индикаторы
            self.feature_columns = ["open", "high", "low", "close", "volume"] + added_feature_cols
            self.logger.info(f"📊 Обновленные feature_columns: {self.feature_columns}")

            logger.info(f"Загружено {len(df)} записей с {df['date'].min()} по {df['date'].max()}")
            logger.info(f"📊 Размер данных: {df.shape} (строк: {df.shape[0]}, колонок: {df.shape[1]})")
            logger.info(f"📋 Список колонок в данных: {list(df.columns)}")

            # Валидация данных (теперь с обновленным self.feature_columns)
            self.validate_data(df)

            # Преобразование и сортировка
            self.logger.info("Преобразование и сортировка данных")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self.logger.info(f"Загружено {len(df)} записей с {df['date'].min()} по {df['date'].max()}")

            # Проверка временного разрешения
            time_diff = df["date"].diff().min()
            self.logger.debug(f"Минимальный интервал данных: {time_diff}")

            # Разделение данных
            self.logger.info("Разделение данных на наборы")
            total = len(df)
            cumulative = 0
            splits = {}

            # Проверка суммы процентов
            total_percent = sum(percentages.values())
            if abs(total_percent - 1.0) > 0.01:
                self.logger.warning(f"Сумма процентов ({total_percent:.2f}) не равна 1.0, корректировка")
                percentages = {k: v / total_percent for k, v in percentages.items()}

            for dataset, percent in percentages.items():
                if percent <= 0:
                    continue

                segment_size = int(total * percent)
                end_idx = min(cumulative + segment_size, total)
                splits[dataset] = (cumulative, end_idx)
                cumulative = end_idx
                self.logger.info(f"{dataset}: {percent:.1%} ({segment_size} записей) [{cumulative}/{total}]")

            # Обработка каждого набора
            for dataset, (start_idx, end_idx) in splits.items():
                if dataset not in output_files:
                    continue

                self.logger.info(f"Обработка набора '{dataset}'")
                segment = df.iloc[start_idx:end_idx].copy()
                self.logger.info(
                    f"Превью данных: \n{tabulate(segment.tail(), headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False)}")

                # Создание окон (теперь с обновленным self.feature_columns)
                windows, keys_map = self.create_windows(segment, dataset)

                # Сохранение
                if windows:
                    self.save_npz(windows, keys_map, output_files[dataset])
                else:
                    self.logger.warning(f"Пропуск сохранения {dataset}: нет данных для окон")

            total_time = time.time() - start_time
            self.logger.success(f"Обработка завершена успешно за {total_time:.2f} секунд")

        except Exception as e:
            self.logger.exception(f"Критическая ошибка при обработке данных: {str(e)}")
            raise


# Пример использования
if __name__ == "__main__":
    logger.info("Запуск Financial Data Preprocessor")

    try:
        preprocessor = FinancialDataPreprocessor(
            ticker=settings.TICKER,
            window_size=150
        )

        # Исправленные проценты (сумма должна быть 1.0)
        # Предположим, что было опечатка в val: 0.5 вместо 0.05
        corrected_percentages = {
            "train": settings.percent_train,        # 75% для обучения
            "val": settings.percent_val,            # 5% для валидации (исправлено)
            "test": settings.percent_test,          # 10% для тестирования
            "backtest": settings.percent_backtest   # 10% для бэктеста
        }

        # Проверка суммы процентов
        total_percent = sum(corrected_percentages.values())
        if abs(total_percent - 1.0) > 0.01:
            logger.warning(f"⚠️ Сумма процентов ({total_percent:.2f}) не равна 1.0")

        preprocessor.process_dataset(
            csv_path=settings.CSV_FILE,
            output_files={
                "train": settings.train_files,
                "val": settings.val_files,
                "test": settings.test_files,
                "backtest": settings.backtest_files
            },
            percentages=corrected_percentages
        )

        logger.success("Все файлы успешно созданы!")

    except Exception as e:
        logger.critical(f"Завершение работы из-за критической ошибки: {str(e)}")
        sys.exit(1)
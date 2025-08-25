# data/csv_to_npz.py

from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Callable
import sys
import time
import talib  # Добавлен импорт TA-Lib
import math
from pathlib import Path
from tabulate import tabulate

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
        self.feature_columns = ["open", "high", "low", "close", "volume"]
        self.logger = logger.bind(component="FinancialPreprocessor")

    def calculate_hma(self, close: pd.Series, timeperiod: int) -> pd.Series:
        """Расчет Hull Moving Average через WMA, так как прямая реализация HMA отсутствует в TA-Lib"""
        self.logger.debug(f"Рассчитываем HMA с периодом {timeperiod} 📐")

        # Формула HMA: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = max(1, int(timeperiod / 2))
        sqrt_period = max(1, int(math.sqrt(timeperiod)))

        wma_half = talib.WMA(close, timeperiod=half_period)
        wma_full = talib.WMA(close, timeperiod=timeperiod)

        # Вычисляем разницу с учетом NaN значений
        diff = 2 * wma_half - wma_full
        hma = talib.WMA(diff, timeperiod=sqrt_period)

        return hma

    @logger.catch
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов в DataFrame"""
        self.logger.info("Добавляем технические индикаторы: MACD, RSI, Bollinger Bands, 2 HMA, ATR 📈")

        # Проверяем наличие колонки 'close'
        if 'close' not in df.columns:
            self.logger.error("Отсутствует колонка 'close' для расчета индикаторов ❌")
            raise ValueError("Колонка 'close' обязательна для расчета технических индикаторов")

        # Сохраняем исходные индексы для последующего восстановления
        original_index = df.index

        # Сбрасываем индекс для корректной работы TA-Lib
        df = df.reset_index(drop=True)

        # Рассчитываем MACD (12,26,9)
        self.logger.debug("Рассчитываем MACD (12,26,9) 📊")
        macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        # Рассчитываем RSI (14)
        self.logger.debug("Рассчитываем RSI (14) 📊")
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)

        # Рассчитываем Bollinger Bands (20, 2)
        self.logger.debug("Рассчитываем Bollinger Bands (20) 📊")
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_mid'] = middle
        df['bb_lower'] = lower

        # Рассчитываем две HMA с разными периодами (9 и 21)
        # Как указано в источниках, HMA не реализован напрямую в TA-Lib и должен рассчитываться через WMA [[6]]
        self.logger.debug("Рассчитываем Hull Moving Average (HMA) с периодами 9 и 21 📐")
        df['hma_fast'] = self.calculate_hma(df['close'], 9)
        df['hma_slow'] = self.calculate_hma(df['close'], 21)


        # Рассчитываем ATR (14)
        self.logger.debug("Рассчитываем Average True Range (ATR) с периодом 14 📐")
        atr = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['atr'] = atr

        # Восстанавливаем исходные индексы
        df.index = original_index
        # заполняем NaN нулями
        df.fillna(0, inplace=True)


        # Обновляем список feature_columns с новыми индикаторами
        new_features = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'bb_upper', 'bb_mid', 'bb_lower', 'hma_fast', 'hma_slow', 'atr']
        self.feature_columns += new_features

        self.logger.success(f"Успешно добавлено {len(new_features)} технических индикаторов! 🎯")
        # Выводим превью данных
        self.logger.info("📊 Превью данных:")

        self.logger.info(f"\n {tabulate(df.head(5), showindex=True, headers='keys', tablefmt='psql')}")
        self.logger.info(f"\n {tabulate(df.tail(5), showindex=True, headers='keys', tablefmt='psql')}")


        return df

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

        # Проверка на наличие пропусков
        null_counts = df[self.feature_columns].isnull().sum()
        total_null = null_counts.sum()

        if total_null > 0:
            self.logger.warning(f"⚠️ Обнаружено {total_null} пропущенных значений")
            for col, count in null_counts.items():
                if count > 0:
                    self.logger.debug(f"🔍 Пропуски в '{col}': {count}")
        else:
            self.logger.success("✅ Нет пропущенных значений в данных")

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
    def process_dataset(self, csv_path: str, output_files: Dict[str, str], percentages: Dict[str, float]) -> None:
        """Основной метод обработки данных"""
        self.logger.info(f"🚀 Начало обработки данных из {csv_path}")
        start_time = time.time()

        try:
            # Загрузка данных
            self.logger.info("📥 Загрузка данных из CSV")
            df = pd.read_csv(csv_path)
            self.logger.info(f"📊 Загружено {len(df)} записей")

            # Добавление технических индикаторов
            df = self.add_technical_indicators(df)

            # Валидация данных
            self.validate_data(df)

            # Преобразование и сортировка
            self.logger.info("🔄 Преобразование и сортировка данных")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self.logger.info(f"📅 Отсортировано {len(df)} записей с {df['date'].min()} по {df['date'].max()}")

            # Проверка временного разрешения
            time_diff = df["date"].diff().min()
            self.logger.debug(f"⏱️ Минимальный интервал данных: {time_diff}")

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
    logger.info("🚀 Запуск Financial Data Preprocessor")

    try:
        # Исправленные проценты (ранее было 0.5 для val, что давало 145%)
        percentages = {
            "train": 0.75,  # 75% для обучения
            "val": 0.05,  # 5% для валидации (было 0.5 - ошибка)
            "test": 0.10,  # 10% для тестирования
            "backtest": 0.10  # 10% для бэктеста
        }

        # Проверка суммы процентов
        total_percent = sum(percentages.values())
        if abs(total_percent - 1.0) > 0.01:
            logger.warning(f"⚠️ Сумма процентов ({total_percent:.2f}) не равна 1.0")

        preprocessor = FinancialDataPreprocessor(
            ticker="DOGEUSDT",
            window_size=150
        )

        preprocessor.process_dataset(
            csv_path="data.csv",
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
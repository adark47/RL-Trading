# data/check_npz.py

from loguru import logger
import numpy as np
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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
    f"{log_dir}/npz_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


class NPZValidator:
    """Класс для валидации файлов .npz, содержащих финансовые временные ряды для ML"""

    def __init__(self, window_size: int = 150):
        self.window_size = window_size
        self.logger = logger.bind(component="NPZValidator")
        self.validation_results = defaultdict(dict)

    def _get_array_preview(self, array: np.ndarray, max_rows: int = 3) -> str:
        """Создает превью массива с использованием tabulate"""
        if array.ndim == 1:
            preview_data = array[:max_rows].reshape(-1, 1)
            headers = ["Value"]
        else:
            preview_data = array[:max_rows, :]
            headers = [f"Feature {i}" for i in range(preview_data.shape[1])]

        return tabulate(preview_data,
                        headers=headers,
                        tablefmt="psql",
                        floatfmt=".4f",
                        showindex="always")

    def validate_file(self, file_path: str) -> Dict:
        """Валидация одного файла .npz"""
        file_path = Path(file_path)
        self.logger.info(f"🔍 Начало валидации файла: {file_path.name} 📁")
        result = {
            "exists": False,
            "valid_structure": False,
            "window_count": 0,
            "window_shape": None,
            "feature_count": 0,
            "contains_nan": False,
            "temporal_consistency": False,
            "key_map_valid": False,
            "date_range": None,
            "ticker": None
        }

        # Проверка существования файла
        if not file_path.exists():
            self.logger.error(f"❌ Файл не найден: {file_path}")
            return result

        result["exists"] = True
        self.logger.success(f"✅ Файл найден: {file_path} ({file_path.stat().st_size / (1024 * 1024):.2f} MB) 📦")

        try:
            # Загрузка файла
            with np.load(file_path, allow_pickle=True) as data:
                # Проверка структуры
                all_keys = list(data.keys())
                array_keys = [k for k in all_keys if k != "_keys_map_"]
                has_key_map = "_keys_map_" in all_keys

                # Проверка наличия ключевой карты
                if not has_key_map:
                    self.logger.error("❌ Отсутствует обязательный ключ '_keys_map_' в файле")
                else:
                    result["key_map_valid"] = True
                    self.logger.info("✅ Найден ключ '_keys_map_' с метаданными 🗺️")

                # Проверка наличия данных
                if not array_keys:
                    self.logger.error("❌ Файл не содержит данных (отсутствуют числовые массивы)")
                    return result

                # Проверка структуры данных
                first_array = data[array_keys[0]]
                window_shape = first_array.shape
                result["window_shape"] = window_shape
                result["feature_count"] = window_shape[1] if len(window_shape) > 1 else 1
                result["window_count"] = len(array_keys)

                self.logger.info(f"📊 Найдено {result['window_count']} окон размером {window_shape}")

                # Проверка формы окон
                inconsistent_shapes = []
                for key in array_keys:
                    if data[key].shape != window_shape:
                        inconsistent_shapes.append((key, data[key].shape))

                if inconsistent_shapes:
                    self.logger.error(f"❌ Обнаружены несоответствия в форме окон: {inconsistent_shapes}")
                else:
                    result["valid_structure"] = True
                    self.logger.success(f"✅ Все окна имеют одинаковую форму: {window_shape} 📐")

                # Проверка на NaN
                contains_nan = False
                nan_count = 0
                for key in array_keys[:10]:  # Проверяем только первые 10 для скорости
                    if np.isnan(data[key]).any():
                        contains_nan = True
                        nan_count += np.isnan(data[key]).sum()

                if contains_nan:
                    self.logger.warning(f"⚠️ Обнаружены NaN значения (примерно {nan_count}+) в данных")
                    result["contains_nan"] = True
                else:
                    self.logger.success("✅ Данные не содержат NaN значений 🧹")

                # Анализ временных меток
                if has_key_map:
                    keys_map = data["_keys_map_"].item() if isinstance(data["_keys_map_"], np.ndarray) else data[
                        "_keys_map_"]
                    dates = []
                    tickers = set()

                    for idx, (ticker, date) in keys_map.items():
                        try:
                            # Пробуем преобразовать в datetime, если это строка
                            if isinstance(date, str):
                                date = pd.to_datetime(date)
                            dates.append(date)
                            tickers.add(ticker)
                        except Exception as e:
                            self.logger.debug(f"⚠️ Проблема с обработкой даты для индекса {idx}: {e}")

                    if dates:
                        dates = sorted(dates)
                        result["date_range"] = (dates[0], dates[-1])
                        result["ticker"] = tickers.pop() if len(tickers) == 1 else "Mixed"

                        # Проверка монотонности
                        is_monotonic = all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))
                        result["temporal_consistency"] = is_monotonic

                        if is_monotonic:
                            self.logger.success(f"✅ Временные метки монотонны: от {dates[0]} до {dates[-1]} ⏱️")
                        else:
                            self.logger.error(
                                "❌ Временные метки НЕ монотонны - возможны пересечения или нарушение порядка")

                        # Превью дат
                        preview_dates = [(i, d) for i, d in list(enumerate(dates))[:3]] + \
                                        [(i, d) for i, d in list(enumerate(dates))[-3:]]
                        dates_table = tabulate(preview_dates,
                                               headers=["Window Index", "Date"],
                                               tablefmt="psql")
                        self.logger.info(f"📅 Превью временных меток:\n{dates_table}")
                    else:
                        self.logger.warning("⚠️ Не удалось извлечь временные метки из _keys_map_")

                # Превью данных
                if array_keys:
                    preview_key = array_keys[0]
                    preview_data = data[preview_key]
                    preview_text = self._get_array_preview(preview_data)
                    self.logger.info(f"🔍 Превью данных (окно {preview_key}):\n{preview_text}")

                # Статистический анализ
                if array_keys:
                    sample = data[array_keys[0]]
                    if sample.ndim > 1:  # Если это многомерные данные
                        stats = {
                            "min": np.min(sample),
                            "max": np.max(sample),
                            "mean": np.mean(sample),
                            "std": np.std(sample)
                        }
                        stats_table = [
                            ["Min", f"{stats['min']:.4f}"],
                            ["Max", f"{stats['max']:.4f}"],
                            ["Mean", f"{stats['mean']:.4f}"],
                            ["Std", f"{stats['std']:.4f}"]
                        ]
                        self.logger.info(f"📊 Статистика по примеру данных:\n{tabulate(stats_table, tablefmt='psql')}")

        except Exception as e:
            self.logger.exception(f"💥 Критическая ошибка при валидации файла: {str(e)}")
            return result

        self.logger.success(f"✅ Валидация файла {file_path.name} завершена успешно! 🎯")
        return result

    def compare_datasets(self, dataset_results: Dict[str, Dict]) -> None:
        """Сравнение различных наборов данных для проверки пересечений и согласованности"""
        self.logger.info("🔍 Сравнение наборов данных для проверки целостности разделения")

        # Проверка пересечений временных интервалов
        date_ranges = {}
        for dataset, result in dataset_results.items():
            if result["date_range"]:
                start, end = result["date_range"]
                date_ranges[dataset] = (start, end)
                self.logger.debug(f"🕒 {dataset} диапазон: {start} - {end}")

        if len(date_ranges) >= 2:
            sorted_datasets = sorted(date_ranges.items(), key=lambda x: x[1][0])

            # Проверка на пересечение временных интервалов
            for i in range(1, len(sorted_datasets)):
                prev_dataset, (prev_start, prev_end) = sorted_datasets[i - 1]
                curr_dataset, (curr_start, curr_end) = sorted_datasets[i]

                if prev_end > curr_start:
                    self.logger.error(
                        f"❌ ПЕРЕСЕЧЕНИЕ: {prev_dataset} заканчивается {prev_end}, но {curr_dataset} начинается {curr_start}")
                else:
                    self.logger.success(f"✅ Нет пересечения между {prev_dataset} и {curr_dataset}")

        # Проверка согласованности размеров окон
        window_shapes = {ds: res["window_shape"] for ds, res in dataset_results.items() if res["window_shape"]}
        if len(set(window_shapes.values())) > 1:
            self.logger.warning(f"⚠️ Разные формы окон между наборами: {window_shapes}")
        elif window_shapes:
            self.logger.success(f"✅ Все наборы имеют одинаковую форму окон: {list(window_shapes.values())[0]}")

        # Проверка количества фич
        feature_counts = {ds: res["feature_count"] for ds, res in dataset_results.items()}
        if len(set(feature_counts.values())) > 1:
            self.logger.error(f"❌ Разное количество фич между наборами: {feature_counts}")
        else:
            self.logger.success(f"✅ Все наборы имеют одинаковое количество фич: {list(feature_counts.values())[0]}")

        # Проверка распределения данных (упрощенная)
        self.logger.info("📊 Проверка распределения данных между наборами")
        total_windows = sum(res["window_count"] for res in dataset_results.values())
        for ds, res in dataset_results.items():
            percent = res["window_count"] / total_windows * 100 if total_windows > 0 else 0
            self.logger.info(f"📈 {ds}: {res['window_count']} окон ({percent:.1f}%)")

    def generate_validation_report(self, file_paths: List[str]) -> None:
        """Генерация полного отчета по всем файлам"""
        self.logger.info("📝 Генерация полного отчета валидации для наборов данных")

        # Валидация каждого файла
        dataset_results = {}
        for file_path in file_paths:
            dataset_name = Path(file_path).stem.split('_')[0]  # Извлекаем имя набора (train, val и т.д.)
            result = self.validate_file(file_path)
            dataset_results[dataset_name] = result
            self.validation_results[dataset_name] = result

        # Сравнение наборов
        self.compare_datasets(dataset_results)

        # Генерация сводного отчета
        summary = []
        for dataset, result in dataset_results.items():
            status = "✅" if all([
                result["exists"],
                result["valid_structure"],
                not result["contains_nan"],
                result["temporal_consistency"],
                result["key_map_valid"]
            ]) else "❌"

            summary.append([
                status,
                dataset,
                "Да" if result["exists"] else "Нет",
                result["window_count"],
                str(result["window_shape"]) if result["window_shape"] else "N/A",
                "Нет" if not result["contains_nan"] else "Да",
                f"{result['date_range'][0]} - {result['date_range'][1]}" if result["date_range"] else "N/A"
            ])

        # Вывод сводной таблицы
        summary_table = tabulate(
            summary,
            headers=["Статус", "Набор", "Существует", "Кол-во окон", "Форма окна", "Есть NaN", "Диапазон дат"],
            tablefmt="grid"
        )
        self.logger.info(f"\n📋 Сводный отчет валидации:\n{summary_table}")

        # Проверка общих рекомендаций
        self._check_best_practices(dataset_results)

        self.logger.success("✅ Полная валидация завершена! Все проверки пройдены. 🏆")

    def _check_best_practices(self, dataset_results: Dict[str, Dict]) -> None:
        """Проверка соблюдения лучших практик валидации финансовых данных"""
        self.logger.info("🔍 Проверка соблюдения лучших практик валидации")

        # Проверка достаточности данных
        total_windows = sum(res["window_count"] for res in dataset_results.values())
        if total_windows < 1000:
            self.logger.warning(
                "⚠️ Общее количество окон меньше 1000 - может быть недостаточно для надежного обучения ML моделей")
        else:
            self.logger.success("✅ Достаточное количество данных для обучения моделей")

        # Проверка разделения данных
        required_datasets = ["train", "val", "test"]
        missing_datasets = [ds for ds in required_datasets if ds not in dataset_results]

        if missing_datasets:
            self.logger.error(f"❌ Отсутствуют обязательные наборы данных: {missing_datasets}")
        else:
            self.logger.success("✅ Все обязательные наборы данных присутствуют (train, val, test)")
            self.logger.info(
                "Data Validation: Verify AI outputs against traditional models during transition periods. [[7]]")

        # Проверка размеров наборов
        if "train" in dataset_results and "val" in dataset_results and "test" in dataset_results:
            train_size = dataset_results["train"]["window_count"]
            val_size = dataset_results["val"]["window_count"]
            test_size = dataset_results["test"]["window_count"]

            total = train_size + val_size + test_size
            if train_size / total < 0.6:
                self.logger.warning(f"⚠️ Доля обучающего набора ({train_size / total:.1%}) ниже рекомендуемых 60-70%")
            if val_size / total < 0.1:
                self.logger.warning(f"⚠️ Доля валидационного набора ({val_size / total:.1%}) ниже рекомендуемых 10-20%")
            if test_size / total < 0.1:
                self.logger.warning(f"⚠️ Доля тестового набора ({test_size / total:.1%}) ниже рекомендуемых 10-20%")
            else:
                self.logger.success("✅ Соотношение наборов соответствует лучшим практикам")

        # Проверка временной целостности
        if len(dataset_results) >= 2:
            date_ranges = {ds: res["date_range"] for ds, res in dataset_results.items() if res["date_range"]}
            if len(date_ranges) == len(dataset_results):
                sorted_ranges = sorted(date_ranges.items(), key=lambda x: x[1][0] if x[1] else pd.Timestamp.min)

                # Проверка временной последовательности
                for i in range(1, len(sorted_ranges)):
                    prev_end = sorted_ranges[i - 1][1][1]
                    curr_start = sorted_ranges[i][1][0]
                    if prev_end >= curr_start:
                        self.logger.error(
                            f"❌ Нарушена временная последовательность между {sorted_ranges[i - 1][0]} и {sorted_ranges[i][0]}")
                    else:
                        self.logger.success(
                            f"✅ Временная последовательность сохранена между {sorted_ranges[i - 1][0]} и {sorted_ranges[i][0]}")

        self.logger.info(
            "Learn important data validation best practices and techniques to improve data integrity for financial planning, forecasting, and budgeting. [[3]]")
        self.logger.info(
            "When validating financial data, ensure temporal consistency to prevent look-ahead bias in models. [[6]]")


if __name__ == "__main__":
    logger.info("🔍 Запуск валидатора NPZ файлов для финансовых данных")

    try:
        # Файлы для валидации
        files_to_validate = [
            "train_data.npz",
            "val_data.npz",
            "test_data.npz",
            "backtest_data.npz"
        ]

        # Проверка существования файлов перед началом
        existing_files = [f for f in files_to_validate if Path(f).exists()]
        missing_files = [f for f in files_to_validate if not Path(f).exists()]

        if missing_files:
            logger.warning(f"⚠️ Отсутствуют файлы: {missing_files}")

        if not existing_files:
            logger.error("❌ Нет файлов для валидации. Проверьте пути к файлам.")
            sys.exit(1)

        # Создание и запуск валидатора
        validator = NPZValidator(window_size=150)
        validator.generate_validation_report(existing_files)

        logger.success("🎉 Валидация завершена успешно! Все проверки пройдены. 🏆")

    except Exception as e:
        logger.exception(f"💥 Критическая ошибка во время валидации: {str(e)}")
        sys.exit(1)
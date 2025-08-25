# data/check_npz.py

import numpy as np
import os
import sys
from loguru import logger
import tabulate
from datetime import datetime

from collections import defaultdict
import warnings
from typing import Dict, Any, List, Optional, Tuple

# Подавляем предупреждения NumPy о сравнении с NaN
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Настройка логгера с цветами и эмодзи
log_dir = "logs"
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"{log_dir}/check_npz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)



class NPZValidator:
    """Класс для валидации .npz файлов и генерации отчетов"""

    def __init__(self):
        self.validation_results = {}
        self.files_to_check = [
            'train_data.npz',
            'val_data.npz',
            'test_data.npz',
            'backtest_data.npz'
        ]
        self.required_arrays = {
            'train_data.npz': ['X_train', 'y_train'],
            'val_data.npz': ['X_val', 'y_val'],
            'test_data.npz': ['X_test', 'y_test'],
            'backtest_data.npz': ['X_backtest', 'y_backtest', 'timestamps']
        }
        self.dataset_types = {
            'train_data.npz': 'Обучающий набор',
            'val_data.npz': 'Валидационный набор',
            'test_data.npz': 'Тестовый набор',
            'backtest_data.npz': 'Бэктест'
        }

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Полная валидация одного файла с детальным анализом"""
        if not os.path.exists(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return {
                'status': 'not_found',
                'errors': [f"Файл отсутствует в директории"],
                'file': file_path
            }

        logger.info(f"Начинаем валидацию: {file_path}")
        results = {
            'status': 'valid',
            'file': file_path,
            'type': self.dataset_types.get(file_path, 'Неизвестный тип'),
            'arrays': {},
            'warnings': [],
            'errors': [],
            'stats': {}
        }

        try:
            # Загружаем файл
            with np.load(file_path, allow_pickle=True) as data:
                # Проверяем наличие обязательных массивов
                required = self.required_arrays.get(file_path, [])
                for array_name in required:
                    if array_name not in data.files:
                        error_msg = f"Отсутствует обязательный массив: {array_name}"
                        results['errors'].append(error_msg)
                        results['status'] = 'invalid'
                        logger.error(error_msg)

                # Анализируем каждый массив в файле
                for key in data.files:
                    array = data[key]
                    shape = array.shape
                    dtype = str(array.dtype)

                    # Собираем информацию о массиве
                    array_info = {
                        'shape': shape,
                        'dtype': dtype,
                        'size': array.size,
                        'ndim': array.ndim,
                        'contains_nan': False,
                        'contains_inf': False,
                        'min': None,
                        'max': None,
                        'mean': None,
                        'std': None
                    }

                    # Проверка на NaN и inf только для числовых типов
                    if np.issubdtype(array.dtype, np.number):
                        try:
                            array_info['contains_nan'] = np.isnan(array).any()
                            array_info['contains_inf'] = np.isinf(array).any()

                            # Статистика только для небольших массивов или первых элементов
                            if array.size > 0:
                                # Для больших массивов берем выборку
                                if array.size > 10000:
                                    # Создаем индексы для случайной выборки
                                    indices = np.random.choice(array.size, size=min(1000, array.size), replace=False)
                                    sample = array.flatten()[indices]
                                else:
                                    sample = array.flatten()

                                # Вычисляем статистику
                                array_info['min'] = float(np.min(sample))
                                array_info['max'] = float(np.max(sample))
                                array_info['mean'] = float(np.mean(sample))
                                array_info['std'] = float(np.std(sample))
                        except Exception as e:
                            logger.debug(f"Не удалось вычислить статистику для {key}: {str(e)}")

                    # Сохраняем информацию
                    results['arrays'][key] = array_info

                    # Проверки на проблемы
                    if array.size == 0:
                        warning = f"Массив '{key}' пустой"
                        results['warnings'].append(warning)
                        logger.warning(warning)

                    if array_info['contains_nan']:
                        error = f"Массив '{key}' содержит NaN значения"
                        results['errors'].append(error)
                        results['status'] = 'invalid'
                        logger.error(error)

                    if array_info['contains_inf']:
                        error = f"Массив '{key}' содержит бесконечные значения"
                        results['errors'].append(error)
                        results['status'] = 'invalid'
                        logger.error(error)

                # Дополнительные проверки для конкретных типов данных
                if 'train' in file_path and 'train_data.npz' == file_path:
                    self._validate_train_data(file_path, data, results)
                elif 'test' in file_path and 'test_data.npz' == file_path:
                    self._validate_test_data(file_path, data, results)
                elif 'backtest' in file_path and 'backtest_data.npz' == file_path:
                    self._validate_backtest_data(file_path, data, results)

                # Статистика по файлу
                results['stats'] = {
                    'total_arrays': len(data.files),
                    'valid_arrays': len(data.files) - len(results['errors']),
                    'has_X': any('X' in key for key in data.files),
                    'has_y': any('y' in key for key in data.files)
                }

                return results

        except Exception as e:
            error_msg = f"Критическая ошибка при обработке файла: {str(e)}"
            results['errors'].append(error_msg)
            results['status'] = 'error'
            logger.exception(f"Ошибка при валидации {file_path}")
            return results

    def _validate_train_data(self, file_path: str, data: np.lib.npyio.NpzFile, results: Dict[str, Any]) -> None:
        """Специфические проверки для обучающих данных"""
        if 'X_train' in data:
            X = data['X_train']
            if X.ndim != 2:
                warning = "X_train должен быть двумерным массивом (образцы × признаки)"
                results['warnings'].append(warning)
                logger.warning(warning)

            if X.size == 0:
                error = "X_train пустой - нет обучающих данных"
                results['errors'].append(error)
                results['status'] = 'invalid'
                logger.error(error)

            # Проверка количества образцов
            if X.shape[0] < 100:
                warning = f"Небольшое количество обучающих образцов: {X.shape[0]} (рекомендуется > 100)"
                results['warnings'].append(warning)
                logger.warning(warning)

    def _validate_test_data(self, file_path: str, data: np.lib.npyio.NpzFile, results: Dict[str, Any]) -> None:
        """Специфические проверки для тестовых данных"""
        # Проверяем соответствие признаков с обучающими данными, если они уже валидированы
        if 'train_data.npz' in self.validation_results:
            train_result = self.validation_results['train_data.npz']
            if train_result.get('status') == 'valid' and 'X_train' in train_result.get('arrays', {}):
                if 'X_test' in data:
                    X_test = data['X_test']
                    train_shape = train_result['arrays']['X_train']['shape']

                    if X_test.ndim >= 2 and train_shape and len(train_shape) >= 2:
                        if X_test.shape[1] != train_shape[1]:
                            error = f"Количество признаков в X_test ({X_test.shape[1]}) не совпадает с X_train ({train_shape[1]})"
                            results['errors'].append(error)
                            results['status'] = 'invalid'
                            logger.error(error)

    def _validate_backtest_data(self, file_path: str, data: np.lib.npyio.NpzFile, results: Dict[str, Any]) -> None:
        """Специфические проверки для бэктеста"""
        if 'timestamps' in data:
            timestamps = data['timestamps']
            if timestamps.ndim != 1:
                warning = "timestamps должен быть одномерным массивом"
                results['warnings'].append(warning)
                logger.warning(warning)

            # Проверка на хронологический порядок
            if timestamps.size > 1 and np.issubdtype(timestamps.dtype, np.datetime64) or np.issubdtype(timestamps.dtype,
                                                                                                       np.number):
                try:
                    is_sorted = np.all(timestamps[:-1] <= timestamps[1:])
                    if not is_sorted:
                        warning = "timestamps не отсортированы по возрастанию"
                        results['warnings'].append(warning)
                        logger.warning(warning)
                except Exception as e:
                    logger.debug(f"Не удалось проверить сортировку timestamps: {str(e)}")

    def validate_all_files(self) -> None:
        """Валидация всех файлов и сохранение результатов"""
        logger.info("Запуск валидации всех файлов данных")
        logger.info("========================================")

        # Сначала валидируем все файлы
        for file in self.files_to_check:
            logger.info(f"\n{'=' * 30} ВАЛИДАЦИЯ: {file} {'=' * 30}")
            self.validation_results[file] = self.validate_file(file)

        # Теперь проверка согласованности между файлами (после того, как все файлы обработаны)
        self._check_consistency()

        # Генерация отчета
        self._generate_detailed_report()

    def _check_consistency(self) -> None:
        """Проверка согласованности данных между разными файлами"""
        logger.info("\nПроверка согласованности данных между наборами...")

        # Проверка признаков между train и другими наборами
        if 'train_data.npz' in self.validation_results and self.validation_results['train_data.npz'].get(
                'status') == 'valid':
            train_info = self.validation_results['train_data.npz']

            if 'X_train' in train_info.get('arrays', {}):
                train_shape = train_info['arrays']['X_train']['shape']

                for file in ['val_data.npz', 'test_data.npz', 'backtest_data.npz']:
                    if file in self.validation_results and self.validation_results[file].get('status') == 'valid':
                        # Определяем имя массива признаков в зависимости от типа файла
                        prefix = file.split('_')[0]
                        x_name = f"X_{prefix}"

                        if x_name in self.validation_results[file]['arrays']:
                            test_shape = self.validation_results[file]['arrays'][x_name]['shape']

                            # Проверяем только если массивы двумерные
                            if len(test_shape) >= 2 and len(train_shape) >= 2:
                                if test_shape[1] != train_shape[1]:
                                    msg = f"Несоответствие признаков: {file} имеет {test_shape[1]} признаков, а train имеет {train_shape[1]}"
                                    self.validation_results[file]['errors'].append(msg)
                                    self.validation_results[file]['status'] = 'invalid'
                                    logger.error(msg)

    def _generate_detailed_report(self) -> None:
        """Генерация детального отчета с использованием tabulate"""
        logger.info("\n\n📊 ДЕТАЛЬНЫЙ ОТЧЕТ ПО ВАЛИДАЦИИ ДАННЫХ")
        logger.info("========================================")

        # 1. Общая сводка по статусам
        self._generate_summary_section()

        # 2. Детальный отчет по каждому файлу
        for file, result in self.validation_results.items():
            self._generate_file_report(file, result)

        # 3. Проверка согласованности между наборами
        self._generate_consistency_report()

        # 4. Итоговые рекомендации
        self._generate_recommendations()

    def _generate_summary_section(self) -> None:
        """Генерация сводной таблицы по всем файлам"""
        logger.info("\n1. ОБЩАЯ СВОДКА ПО ВАЛИДАЦИИ:")

        summary_data = []
        for file, result in self.validation_results.items():
            status = result.get('status', 'error')
            if status == 'valid':
                status_display = "✅ Валиден"
                status_color = "success"
            elif status == 'invalid':
                status_display = "❌ Невалиден"
                status_color = "error"
            elif status == 'not_found':
                status_display = "⚠️ Не найден"
                status_color = "warning"
            else:
                status_display = "🐞 Ошибка"
                status_color = "error"

            # Подсчет ошибок и предупреждений
            error_count = len(result.get('errors', []))
            warning_count = len(result.get('warnings', []))

            summary_data.append([
                file,
                status_display,
                self.dataset_types.get(file, 'N/A'),
                f"{result.get('stats', {}).get('total_arrays', 0)}",
                f"{'❌ ' + str(error_count) if error_count > 0 else '✅ 0'}",
                f"{'⚠️ ' + str(warning_count) if warning_count > 0 else '0'}"
            ])

        # Вывод сводной таблицы
        headers = ["Файл", "Статус", "Тип набора", "Массивы", "Ошибки", "Предупр."]
        table = tabulate.tabulate(
            summary_data,
            headers=headers,
            tablefmt="rounded_grid",
            stralign="center"
        )
        logger.opt(colors=True).info(f"\n{table}")

    def _generate_file_report(self, file: str, result: Dict[str, Any]) -> None:
        """Генерация детального отчета для одного файла"""
        if result.get('status') == 'not_found':
            logger.warning(f"\n\n⚠️ ФАЙЛ {file} НЕ НАЙДЕН")
            logger.info("  • Проверьте путь к файлу и его наличие в рабочей директории")
            return

        status = result.get('status', 'error')
        status_emoji = "✅" if status == 'valid' else "❌"
        logger.info(f"\n\n{status_emoji} ДЕТАЛЬНЫЙ ОТЧЕТ: {file} ({self.dataset_types.get(file, 'N/A')})")

        # Информация о статусе
        if status == 'valid':
            logger.success("✓ Статус: Все проверки пройдены успешно")
        else:
            logger.error(f"✗ Статус: {len(result.get('errors', []))} критических ошибок")

        # Основная информация о файле
        logger.info("\n📌 ОСНОВНАЯ ИНФОРМАЦИЯ:")
        stats = result.get('stats', {})
        logger.info(f"  • Общее количество массивов: {stats.get('total_arrays', 0)}")
        logger.info(f"  • Наличие признаков (X): {'Да' if stats.get('has_X') else 'Нет'}")
        logger.info(f"  • Наличие целевой переменной (y): {'Дa' if stats.get('has_y') else 'Нет'}")

        # Проверка обязательных массивов
        required = self.required_arrays.get(file, [])
        missing = []
        if 'arrays' in result:
            missing = [arr for arr in required if arr not in result['arrays']]

        if missing:
            logger.warning(f"  • Отсутствуют обязательные массивы: {', '.join(missing)}")
        else:
            logger.success("  • Все обязательные массивы присутствуют")

        # Отображение информации о каждом массиве
        if result.get('arrays'):
            logger.info("\n📋 ИНФОРМАЦИЯ О МАССИВАХ:")
            array_data = []
            for name, info in result['arrays'].items():
                # Определяем статус массива
                status = "✅"
                if info.get('contains_nan'):
                    status = "❌ NaN"
                elif info.get('contains_inf'):
                    status = "❌ Inf"
                elif info.get('size', 0) == 0:
                    status = "⚠️ Пустой"

                # Формируем строку с информацией
                shape_str = "×".join(map(str, info['shape'])) if info['shape'] else "0"
                dtype_str = info['dtype']
                size_str = f"{info['size']:,}".replace(',', ' ')

                # Добавляем статистику, если доступна
                stats_str = ""
                if info.get('min') is not None and info.get('max') is not None:
                    stats_str = f"min={info['min']:.2f}, max={info['max']:.2f}"

                array_data.append([
                    name,
                    status,
                    shape_str,
                    dtype_str,
                    size_str,
                    stats_str
                ])

            # Выводим таблицу с информацией о массивах
            headers = ["Имя", "Статус", "Форма", "Тип", "Размер", "Статистика"]
            table = tabulate.tabulate(
                array_data,
                headers=headers,
                tablefmt="simple_outline",
                stralign="left",
                maxcolwidths=[15, 8, 15, 10, 10, 30]
            )
            logger.info(f"\n{table}")

        # Отображение ошибок
        if result.get('errors'):
            logger.error(f"\n🚨 КРИТИЧЕСКИЕ ОШИБКИ ({len(result['errors'])}):")
            for i, error in enumerate(result['errors'], 1):
                logger.error(f"  {i}. {error}")

        # Отображение предупреждений
        if result.get('warnings'):
            logger.warning(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ ({len(result['warnings'])}):")
            for i, warning in enumerate(result['warnings'], 1):
                logger.warning(f"  {i}. {warning}")

        # Превью данных для числовых массивов
        self._generate_data_preview(file, result)

    def _generate_data_preview(self, file: str, result: Dict[str, Any]) -> None:
        """Генерация превью данных для числовых массивов"""
        # Ищем подходящий массив для превью (предпочтительно X_*)
        preview_array = None
        preview_name = None

        # Сначала пробуем загрузить файл
        try:
            with np.load(file, allow_pickle=True) as data:
                for name in data.files:
                    if 'X' in name and data[name].ndim == 2 and data[name].size > 0:
                        preview_array = data[name]
                        preview_name = name
                        break

                # Если не нашли X_*, пробуем найти любой 2D массив
                if preview_array is None:
                    for name in data.files:
                        if data[name].ndim == 2 and data[name].size > 0:
                            preview_array = data[name]
                            preview_name = name
                            break

                if preview_array is not None and preview_array.size > 0:
                    logger.info(f"\n🔍 ПРЕВЬЮ ДАННЫХ ({preview_name}):")

                    # Берем первые 5 строк и первые 4 столбца
                    rows = min(5, preview_array.shape[0])
                    cols = min(4, preview_array.shape[1])
                    sample = preview_array[:rows, :cols]

                    # Форматируем данные для лучшей читаемости
                    formatted_data = []
                    for i in range(rows):
                        row = []
                        for x in sample[i]:
                            if isinstance(x, (float, np.floating)):
                                row.append(f"{x:.4f}")
                            else:
                                row.append(str(x))
                        formatted_data.append(row)

                    # Создаем заголовки
                    headers = [f"Col {i}" for i in range(cols)]

                    # Выводим таблицу
                    table = tabulate.tabulate(
                        formatted_data,
                        headers=headers,
                        tablefmt="rounded_outline",
                        stralign="right"
                    )
                    logger.info(f"\n{table}")

                    if rows < preview_array.shape[0] or cols < preview_array.shape[1]:
                        logger.info(f"\n  • Отображено {rows} строк × {cols} столбцов")
                        logger.info(f"  • Общая форма данных: {preview_array.shape}")
        except Exception as e:
            logger.debug(f"Не удалось сгенерировать превью данных для {file}: {str(e)}")

    def _generate_consistency_report(self) -> None:
        """Генерация отчета о согласованности данных между наборами"""
        logger.info("\n\n🔗 ПРОВЕРКА СОГЛАСОВАННОСТИ МЕЖДУ НАБОРАМИ ДАННЫХ")
        logger.info("========================================")

        # Проверка признаков между train и другими наборами
        if 'train_data.npz' in self.validation_results:
            train_res = self.validation_results['train_data.npz']
            if train_res.get('status') == 'valid' and 'X_train' in train_res.get('arrays', {}):
                train_shape = train_res['arrays']['X_train']['shape']
                logger.success(f"✓ Обучающий набор: {train_shape[0]} образцов, {train_shape[1]} признаков")

                # Сравниваем с другими наборами
                comparisons = []
                for file in ['val_data.npz', 'test_data.npz', 'backtest_data.npz']:
                    if file in self.validation_results:
                        file_res = self.validation_results[file]
                        if file_res.get('status') == 'valid':
                            # Определяем имя массива признаков
                            prefix = file.split('_')[0]
                            x_name = f"X_{prefix}"

                            if x_name in file_res.get('arrays', {}):
                                test_shape = file_res['arrays'][x_name]['shape']

                                if len(test_shape) >= 2 and len(train_shape) >= 2:
                                    if test_shape[1] == train_shape[1]:
                                        comparisons.append([
                                            self.dataset_types[file],
                                            f"{test_shape[0]}",
                                            f"{test_shape[1]}",
                                            "✅ Совпадает"
                                        ])
                                    else:
                                        comparisons.append([
                                            self.dataset_types[file],
                                            f"{test_shape[0]}",
                                            f"{test_shape[1]}",
                                            f"❌ Не совпадает ({train_shape[1]} в train)"
                                        ])

                # Выводим таблицу сравнения
                if comparisons:
                    headers = ["Набор данных", "Образцы", "Признаки", "Согласованность"]
                    table = tabulate.tabulate(
                        comparisons,
                        headers=headers,
                        tablefmt="rounded_grid",
                        stralign="center"
                    )
                    logger.info(f"\nСравнение с обучающим набором:\n{table}")

        # Проверка соотношения размеров
        sizes = {}
        for file in ['train_data.npz', 'val_data.npz', 'test_data.npz']:
            if file in self.validation_results and self.validation_results[file].get('status') == 'valid':
                # Определяем имя массива
                prefix = file.split('_')[0]
                x_name = f"X_{prefix}"

                if x_name in self.validation_results[file].get('arrays', {}):
                    shape = self.validation_results[file]['arrays'][x_name]['shape']
                    sizes[file] = shape[0]

        if len(sizes) >= 2:
            logger.info("\n📊 СООТНОШЕНИЕ РАЗМЕРОВ НАБОРОВ:")
            total = sum(sizes.values())
            ratios = []
            for file, size in sizes.items():
                ratio = (size / total) * 100
                set_type = self.dataset_types[file].replace(" набор", "")

                # Проверяем, соответствует ли соотношение рекомендуемым значениям
                if "test" in file and (10 <= ratio <= 20):
                    status = "✅"
                elif "val" in file and (10 <= ratio <= 20):
                    status = "✅"
                elif "train" in file and (60 <= ratio <= 80):
                    status = "✅"
                else:
                    status = "⚠️"

                ratios.append([set_type, f"{size:,}".replace(',', ' '), f"{ratio:.1f}% ({status})"])

            headers = ["Тип", "Размер", "Доля от общего"]
            table = tabulate.tabulate(
                ratios,
                headers=headers,
                tablefmt="rounded_outline",
                stralign="center"
            )
            logger.info(f"\n{table}")

    def _generate_recommendations(self) -> None:
        """Генерация рекомендаций на основе результатов валидации"""
        logger.info("\n\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ ДАННЫХ")
        logger.info("========================================")

        # Проверка наличия всех необходимых наборов
        required_sets = ['train_data.npz', 'val_data.npz', 'test_data.npz']
        missing_sets = [s for s in required_sets if self.validation_results.get(s, {}).get('status') != 'valid']

        if missing_sets:
            logger.warning(f"⚠️ Отсутствуют или повреждены необходимые наборы данных: {', '.join(missing_sets)}")
            logger.info("  • Рекомендуется создать все три набора: обучающий, валидационный и тестовый")
        else:
            logger.success("✓ Все необходимые наборы данных присутствуют и валидны")

        # Проверка на утечку данных
        has_timestamps = 'backtest_data.npz' in self.validation_results and 'timestamps' in self.validation_results[
            'backtest_data.npz'].get('arrays', {})
        if has_timestamps and 'train_data.npz' in self.validation_results:
            logger.info("\n🔍 Проверка на утечку данных:")
            logger.info(
                "  • Убедитесь, что временные метки в обучающем наборе предшествуют временным меткам в тестовом наборе")
            logger.info("  • Для временных рядов критически важно соблюдать хронологический порядок разделения данных")

        # Проверка баланса классов
        if 'train_data.npz' in self.validation_results:
            train_res = self.validation_results['train_data.npz']
            if 'y_train' in train_res.get('arrays', {}):
                try:
                    with np.load('train_data.npz', allow_pickle=True) as data:
                        if 'y_train' in data:
                            y = data['y_train']
                            # Проверяем, является ли задача классификацией (мало уникальных значений)
                            if np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 20:
                                unique, counts = np.unique(y, return_counts=True)
                                balance = counts / np.sum(counts)

                                logger.info("\n⚖️ Баланс классов в обучающем наборе:")
                                class_data = []
                                for cls, count, pct in zip(unique, counts, balance):
                                    status = "✅" if 0.2 <= pct <= 0.8 else "⚠️"
                                    class_data.append([f"Класс {cls}", f"{count}", f"{pct:.1%} ({status})"])

                                headers = ["Класс", "Количество", "Доля"]
                                table = tabulate.tabulate(
                                    class_data,
                                    headers=headers,
                                    tablefmt="simple",
                                    stralign="center"
                                )
                                logger.info(f"\n{table}")
                                logger.info(
                                    "  • Для задач классификации рекомендуется баланс классов в пределах 20-80%")
                except Exception as e:
                    logger.debug(f"Не удалось проверить баланс классов: {str(e)}")

        # Общие рекомендации
        logger.info("\n📌 ОБЩИЕ РЕКОМЕНДАЦИИ:")
        logger.info("  • Перед обучением убедитесь в отсутствии утечки данных между наборами")
        logger.info("  • Для числовых признаков примените нормализацию на основе обучающего набора")
        logger.info("  • Для временных рядов убедитесь в правильной последовательности данных")
        logger.info("  • Проверьте отсутствие дубликатов в обучающем и тестовом наборах")
        logger.info("  • Убедитесь, что тестовый набор репрезентативен для реальных данных")


def main() -> None:
    """Основная функция для запуска валидации"""
    # Проверка наличия необходимых библиотек
    try:
        import loguru
        import tabulate
    except ImportError:
        logger.warning("Установка необходимых библиотек loguru и tabulate...")
        try:
            os.system("pip install loguru tabulate --quiet")
            from loguru import logger
            import tabulate
        except Exception as e:
            logger.error(f"Не удалось установить необходимые библиотеки: {str(e)}")
            return

    # Запуск валидации
    validator = NPZValidator()
    validator.validate_all_files()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\nПрограмма прервана пользователем (Ctrl+C)")
    except Exception as e:
        logger.exception("Критическая ошибка в программе")
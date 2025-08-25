# data/read_npz.py

import numpy as np
import os
import sys
from loguru import logger
import tabulate

# Настройка логгера с цветами для разных уровней
logger.remove()  # Удаляем стандартный обработчик
logger.add(sys.stderr,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
           colorize=True,
           level="INFO")


def inspect_npz_file(file_path, max_rows=10, max_cols=5, show_all_columns=False):
    """Просматривает содержимое .npz файла и выводит первые элементы массивов"""
    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не найден: {file_path}")
        return False

    try:
        # Загружаем файл с поддержкой объектов
        data = np.load(file_path, allow_pickle=True)

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Файл: {file_path}")
        logger.info(f"{'=' * 50}")

        # Проверяем, есть ли данные в файле
        if len(data.files) == 0:
            logger.warning("⚠️  Файл пустой, массивы не найдены")
            data.close()
            return True

        # Обрабатываем каждый массив в файле
        for key in data.files:
            array = data[key]
            shape = array.shape

            logger.success(f"\nМассив: '{key}'")
            logger.info(f"  • Форма данных: {shape}")
            logger.info(f"  • Тип данных: {array.dtype}")
            logger.info(f"  • Размер: {array.size} элементов")

            # Случай 1: Пустой массив
            if array.size == 0:
                logger.warning("  • Содержимое: пустой массив")
                continue

            # Случай 2: Одномерный массив
            if array.ndim == 1:
                sample = array[:max_rows]
                logger.info(f"  • Первые {min(max_rows, len(sample))} элементов:")
                logger.info(f"    {sample}")

            # Случай 3: Двумерный массив (наиболее вероятный для данных)
            elif array.ndim == 2:
                rows = min(max_rows, shape[0])

                # Определяем, сколько столбцов показывать
                if show_all_columns:
                    cols = shape[1]
                    logger.info(f"  • Отображение всех {cols} столбцов (может быть длинным)")
                else:
                    cols = min(max_cols, shape[1])

                logger.info(f"  • Первые {rows} строк x {cols} столбцов:")

                # Подготовка данных для tabulate
                sample = array[:rows, :cols]

                # Форматируем числа для лучшей читаемости
                formatted_data = []
                for i in range(rows):
                    row = []
                    for x in sample[i]:
                        if isinstance(x, (float, np.floating)):
                            row.append(f"{x:.4f}")
                        else:
                            row.append(str(x))
                    formatted_data.append(row)

                # Добавляем заголовки
                headers = [f"Col {i}" for i in range(cols)]

                # Настройка ширины столбцов
                max_col_width = 12  # Максимальная ширина каждого столбца
                col_widths = [max_col_width] * cols

                # Выводим таблицу с контролируемой шириной столбцов
                table = tabulate.tabulate(formatted_data,
                                          headers=headers,
                                          tablefmt="grid",
                                          stralign="right",
                                          maxcolwidths=col_widths)  # Устанавливаем максимальную ширину для каждого столбца [[7]]
                logger.info(f"\n{table}")

                if not show_all_columns and rows < shape[0]:
                    logger.info(f"    ... и еще {shape[0] - rows} строк")
                if not show_all_columns and cols < shape[1]:
                    logger.warning(
                        f"    ⚠️  Отображено только {cols} из {shape[1]} столбцов. Используйте show_all_columns=True для просмотра всех столбцов.")
                elif show_all_columns and shape[1] > 20:
                    logger.info(
                        f"    💡 Совет: Для очень широких таблиц используйте max_cols параметр для ограничения отображаемых столбцов")

            # Случай 4: Многомерный массив (3D+)
            else:
                logger.info(f"  • Многомерный массив ({array.ndim}D)")
                logger.warning("  • Показываем срез по первому измерению:")

                # Создаем срез первых элементов
                slice_obj = tuple([slice(0, min(2, array.shape[0]))] +
                                  [slice(None) for _ in range(1, array.ndim)])
                sample = array[slice_obj]

                logger.info(f"  • Пример данных (упрощенный):")
                logger.info(f"    Форма среза: {sample.shape}")
                logger.info(f"    Содержимое: {str(sample).replace(chr(10), ' ')}")

        data.close()
        return True

    except Exception as e:
        logger.exception(f"❌ Ошибка при обработке файла {file_path}")
        return False


def main():
    """Основная функция для проверки всех файлов"""
    logger.info("🔍 Начинаем проверку файлов данных...")
    logger.info("========================================")

    files_to_check = [
        'train_data.npz',
        'val_data.npz',
        'test_data.npz',
        'backtest_data.npz'
    ]

    found_files = 0
    for file in files_to_check:
        # Показываем все столбцы только для небольших таблиц, иначе используем ограничение
        if inspect_npz_file(file, max_cols=15, show_all_columns=True):
            found_files += 1

    # Итоговая статистика
    logger.info("\n\n📊 Результаты проверки:")
    if found_files > 0:
        logger.success(f"  • Найдено файлов: {found_files}/{len(files_to_check)}")
    if len(files_to_check) - found_files > 0:
        logger.warning(f"  • Пропущено файлов: {len(files_to_check) - found_files}")
    else:
        logger.success(f"  • Пропущено файлов: {len(files_to_check) - found_files}")

    if found_files > 0:
        logger.info("\n💡 Советы по интерпретации данных:")
        logger.info("  • 'X_*' обычно содержит признаки (фичи)")
        logger.info("  • 'y_*' обычно содержит целевые переменные")
        logger.info("  • Для временных рядов первое измерение - временные шаги")
        logger.info("  • Используйте np.load(file)['key'].shape для проверки формы")
        logger.info("  • Для просмотра всех столбцов используйте параметр show_all_columns=True")
        logger.info("  • Для контроля ширины столбцов используется параметр maxcolwidths [[3]]")


if __name__ == "__main__":
    # Проверяем наличие необходимых библиотек и устанавливаем при необходимости
    try:
        import loguru
        import tabulate
    except ImportError:
        logger.warning("Установка необходимых библиотек loguru и tabulate...")
        os.system("pip install loguru tabulate --quiet")
        from loguru import logger
        import tabulate

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\nПрограмма прервана пользователем (Ctrl+C)")
    except Exception as e:
        logger.exception("Критическая ошибка в программе")
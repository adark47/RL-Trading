import numpy as np
import os
from termcolor import colored


def inspect_npz_file(file_path, max_rows=10, max_cols=5):
    """Просматривает содержимое .npz файла и выводит первые элементы массивов"""
    if not os.path.exists(file_path):
        print(colored(f"❌ Файл не найден: {file_path}", "red"))
        return False

    try:
        # Загружаем файл с поддержкой объектов
        data = np.load(file_path, allow_pickle=True)

        print(colored(f"\n{'=' * 50}", "blue"))
        print(colored(f"Файл: {file_path}", "blue", attrs=["bold"]))
        print(colored(f"{'=' * 50}", "blue"))

        # Проверяем, есть ли данные в файле
        if len(data.files) == 0:
            print(colored("⚠️  Файл пустой, массивы не найдены", "yellow"))
            data.close()
            return True

        # Обрабатываем каждый массив в файле
        for key in data.files:
            array = data[key]
            shape = array.shape

            print(colored(f"\nМассив: '{key}'", "green", attrs=["bold"]))
            print(colored(f"  • Форма данных: {shape}", "cyan"))
            print(colored(f"  • Тип данных: {array.dtype}", "cyan"))
            print(colored(f"  • Размер: {array.size} элементов", "cyan"))

            # Случай 1: Пустой массив
            if array.size == 0:
                print(colored("  • Содержимое: пустой массив", "yellow"))
                continue

            # Случай 2: Одномерный массив
            if array.ndim == 1:
                sample = array[:max_rows]
                print(colored(f"  • Первые {min(max_rows, len(sample))} элементов:", "magenta"))
                print(f"    {sample}")

            # Случай 3: Двумерный массив (наиболее вероятный для данных)
            elif array.ndim == 2:
                rows = min(max_rows, shape[0])
                cols = min(max_cols, shape[1])

                print(colored(f"  • Первые {rows} строк x {cols} столбцов:", "magenta"))

                # Форматируем вывод для лучшей читаемости
                for i in range(rows):
                    row = array[i, :cols]
                    row_str = ", ".join([f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x) for x in row])
                    if shape[1] > max_cols:
                        row_str += f", ... (и еще {shape[1] - max_cols} колонок)"
                    print(f"    Строка {i}: [{row_str}]")

                if rows < shape[0]:
                    print(colored(f"    ... и еще {shape[0] - rows} строк", "yellow"))

            # Случай 4: Многомерный массив (3D+)
            else:
                print(colored(f"  • Многомерный массив ({array.ndim}D)", "magenta"))
                print(colored("  • Показываем срез по первому измерению:", "yellow"))

                # Создаем срез первых элементов
                slice_obj = tuple([slice(0, min(2, array.shape[0]))] +
                                  [slice(None) for _ in range(1, array.ndim)])
                sample = array[slice_obj]

                print(colored(f"  • Пример данных (упрощенный):", "yellow"))
                print(f"    Форма среза: {sample.shape}")
                print(f"    Содержимое: {str(sample).replace(chr(10), ' ')}")

        data.close()
        return True

    except Exception as e:
        print(colored(f"❌ Ошибка при обработке файла {file_path}: {str(e)}", "red"))
        return False


def main():
    """Основная функция для проверки всех файлов"""
    files_to_check = [
        'train_data.npz',
        'val_data.npz',
        'test_data.npz',
        'backtest_data.npz'
    ]

    print(colored("🔍 Начинаем проверку файлов данных...", "cyan", attrs=["bold"]))
    print(colored("========================================", "cyan"))

    found_files = 0
    for file in files_to_check:
        if inspect_npz_file(file):
            found_files += 1

    # Итоговая статистика
    print(colored("\n\n📊 Результаты проверки:", "cyan", attrs=["bold"]))
    print(colored(f"  • Найдено файлов: {found_files}/{len(files_to_check)}", "green"))
    print(colored(f"  • Пропущено файлов: {len(files_to_check) - found_files}",
                  "yellow" if len(files_to_check) - found_files > 0 else "green"))

    if found_files > 0:
        print(colored("\n💡 Советы по интерпретации данных:", "blue"))
        print(colored("  • 'X_*' обычно содержит признаки (фичи)", "cyan"))
        print(colored("  • 'y_*' обычно содержит целевые переменные", "cyan"))
        print(colored("  • Для временных рядов первое измерение - временные шаги", "cyan"))
        print(colored("  • Используйте np.load(file)['key'].shape для проверки формы", "cyan"))


if __name__ == "__main__":
    try:
        # Проверяем наличие необходимых библиотек
        try:
            import termcolor
        except ImportError:
            print(colored("Установка termcolor для цветного вывода...", "yellow"))
            os.system("pip install termcolor --quiet")

        main()
    except KeyboardInterrupt:
        print(colored("\n\nПрограмма прервана пользователем (Ctrl+C)", "yellow"))
    except Exception as e:
        print(colored(f"Критическая ошибка: {str(e)}", "red"))
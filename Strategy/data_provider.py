# data_provider.py

import sys
import os
import pandas as pd
import datetime
# Предполагается, что ArcticDB уже установлен и импортируется как adb
import arcticdb as adb
from tabulate import tabulate
from nautilus_trader import TEST_DATA_DIR
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Import loguru logger
from loguru import logger

# Настройка логгера с цветами и эмоджи
log_dir = "logs"
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"{log_dir}/data_provider_from_arcticDB_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Импортируем конфигурацию

from config import DataPreprocessingConfig as DataPreprocessingConfig

# Создаем экземпляр конфигурации
settings = DataPreprocessingConfig()


def prepare_data_1min():
    """
    Подготавливает данные, считанные из ArcticDB, для использования в Nautilus Trader.
    """
    try:
        logger.info("🚀 Начало подготовки данных из ArcticDB")

        # === ИНИЦИАЛИЗАЦИЯ ARCTICDB ===
        # Используем путь к хранилищу из конфигурации
        # storage_path = settings.ARCTIC_PATH # Предполагаем, что это просто имя папки, например, 'arctic_db'

        # Определяем абсолютный путь к директории data, находящейся на том же уровне, что и директория скрипта
        # __file__ - путь к текущему файлу (data_provider.py)
        # os.path.dirname(__file__) - путь к директории data_provider
        # os.path.dirname(os.path.dirname(__file__)) - путь к родительской директории проекта (предположим, это корень проекта)
        # os.path.join(..., 'data') - путь к директории data внутри корня проекта
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        # Полный путь к хранилищу ArcticDB внутри директории data
        arctic_db_path = os.path.join(data_dir, settings.ARCTIC_PATH)

        logger.debug(f"📁 Путь к ArcticDB: {arctic_db_path}")

        # Инициализируем подключение к ArcticDB, используя абсолютный путь
        # lmdb:// требует указания пути к директории хранилища
        ac = adb.Arctic(f"lmdb://{arctic_db_path}")

        logger.success("🔗 Подключение к ArcticDB установлено")

        # Получаем библиотеку из конфигурации
        library_name = settings.LIBRARY_NAME
        logger.debug(f"📚 Проверка наличия библиотеки: {library_name}")

        if not ac.has_library(library_name):
            error_msg = f"❌ Библиотека {library_name} не найдена в хранилище {arctic_db_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        library = ac.get_library(library_name)
        logger.success(f"📚 Библиотека {library_name} получена")

        # Формируем имя символа, как в get_data_to_arcticDB.py
        symbol_name = f"{settings.TICKER}_{settings.TIMEFRAME}_{settings.MARKET_TYPE}"
        logger.debug(f"📊 Проверка наличия символа: {symbol_name}")

        # Проверяем наличие символа
        if not library.has_symbol(symbol_name):
            error_msg = f"❌ Символ {symbol_name} не найден в библиотеке {library_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # === ЧТЕНИЕ ДАННЫХ ИЗ ARCTICDB ===
        logger.info(f"📥 Чтение данных для символа: {symbol_name}")

        # Определение временного диапазона, если задан DAYS_BACK
        if hasattr(settings, 'DAYS_BACK') and settings.DAYS_BACK > 0:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=settings.DAYS_BACK)
            logger.info(f"📅 Запрашиваем данные за период: {start_date} - {end_date}")

            # Загрузка данных с фильтрацией по времени
            # В ArcticDB 2.x используем кортеж (start, end) вместо DateRange
            try:
                # Загружаем все данные без фильтрации сначала
                logger.debug("🔍 Загружаем данные из ArcticDB...")
                arctic_result = library.read(symbol_name)
                df = arctic_result.data

                # Фильтруем по временному диапазону
                if 'date' in df.columns:
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    logger.success(f"✅ Отфильтровано {len(df)} записей за {settings.DAYS_BACK} дней")
                else:
                    # Если 'date' является индексом, сбросим его для фильтрации
                    df_reset = df.reset_index()
                    if df_reset.columns[0].lower() == 'date' or 'date' in df_reset.columns[0].lower():
                        df_reset = df_reset.rename(columns={df_reset.columns[0]: 'date'})
                        df = df_reset[(df_reset['date'] >= start_date) & (df_reset['date'] <= end_date)]
                        logger.success(f"✅ Отфильтровано {len(df)} записей за {settings.DAYS_BACK} дней")
                    else:
                        logger.error("❌ В данных отсутствует колонка 'date'")
                        raise ValueError("Данные не содержат колонку 'date'")
            except Exception as e:
                logger.error(f"🔥 Ошибка фильтрации данных по дате: {e}")
                raise
        else:
            # Читаем данные из ArcticDB без фильтрации
            arctic_result = library.read(symbol_name)
            df = arctic_result.data
            logger.info("📥 Все данные загружены без фильтрации по дате")

        logger.success(f"📥 Данные успешно прочитаны. Форма: {df.shape}")

        # Убедимся, что индекс - это столбец 'date' в нужном формате
        # (предполагается, что get_data_to_arcticDB.py сохраняет df с 'date' как обычным столбцом)
        # Если 'date' уже индекс, этот шаг может быть не нужен или требовать изменения
        # if 'date' not in df.columns: # Более явная проверка
        #     # Если 'date' является индексом, сбросим его
        #     df = df.reset_index()
        #     # Если после reset_index() имя колонки времени не 'date', переименуем его
        #     # Это маловероятно, если 'date' был индексом, но на всякий случай:
        #     # if df.columns[0] != 'date':
        #     #     df = df.rename(columns={df.columns[0]: 'date'})

        # Проверим, есть ли столбец 'date'. Если его нет, возможно, он является индексом.
        if 'date' not in df.columns:
            logger.warning("📅 Столбец 'date' не найден. Попытка сброса индекса...")
            # Попробуем сбросить индекс, предполагая, что он содержит дату
            df_reset = df.reset_index()
            # Проверим, стал ли первый столбец (бывший индекс) похож на дату
            # Простая проверка: если он называется 'date' или содержит слово 'date'
            if df_reset.columns[0].lower() == 'date' or 'date' in df_reset.columns[0].lower():
                df = df_reset
                df = df.rename(columns={df.columns[0]: 'date'})
                logger.success("📅 Столбец 'date' восстановлен после сброса индекса")
            else:
                # Если не похоже на 'date', оставляем как есть и надеемся, что 'date' появится
                # Или можно вызвать ошибку
                error_msg = "❌ Столбец 'date' не найден в DataFrame после чтения из ArcticDB."
                logger.error(error_msg)
                raise KeyError(error_msg)

        # Убедимся, что столбцы в правильном порядке и названиях
        # get_data_to_arcticDB.py сохраняет: ['date', 'open', 'high', 'low', 'close', 'volume']
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        # Проверим, что все нужные столбцы присутствуют
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            error_msg = f"❌ В DataFrame отсутствуют следующие ожидаемые столбцы: {missing_cols}"
            logger.error(error_msg)
            raise KeyError(error_msg)

        # Переупорядочим столбцы, если нужно
        df = df.reindex(columns=expected_columns)
        logger.debug("📋 Столбцы переупорядочены")

        # Устанавливаем 'date' как индекс, как это делалось с CSV
        logger.info("⏰ Преобразование столбца 'date' в timestamp...")
        df["timestamp"] = pd.to_datetime(df["date"],
                                         errors='coerce')  # Добавим errors='coerce' на случай проблем с форматом
        if df["timestamp"].isna().any():
            logger.warning(
                "⚠️ Внимание: Некоторые значения дат не удалось преобразовать. Проверьте формат столбца 'date'.")
        df = df.drop(columns=["date"])  # Удаляем старый столбец 'date'
        df = df.set_index("timestamp")
        # Сортируем по индексу (времени), на случай, если данные в ArcticDB не отсортированы
        df = df.sort_index()

        logger.success("⏰ Преобразование timestamp завершено")

        # === ПОДГОТОВКА ДАННЫХ ДЛЯ NAUTILUS TRADER ===
        logger.info("🧮 Начало подготовки данных для Nautilus Trader")

        # Define exchange name (ВАЖНО: Проверьте, соответствует ли это логике Nautilus)
        # VENUE_NAME = settings.TICKER # Использование TICKER как имени площадки может быть некорректно
        # Возможно, нужно определить VENUE_NAME отдельно или использовать фиксированное значение
        # Для примера оставим как есть, но это потенциальное место для ошибки
        VENUE_NAME = settings.TICKER  # Или settings.VENUE_NAME если добавите в конфиг

        # Instrument definition
        # ВАЖНО: Убедитесь, что тип инструмента соответствует данным из ArcticDB
        # TestInstrumentProvider.eurusd_future - это пример, возможно, нужен другой провайдер
        # или создание инструмента вручную. Проверьте документацию Nautilus Trader.
        # expiry_year и expiry_month должны соответствовать вашему контракту или быть подходящими фиктивными значениями
        # Для спота или perpetual, возможно, нужно использовать другой провайдер или создавать инструмент иначе.
        logger.debug("🔧 Создание инструмента...")
        _INSTRUMENT = TestInstrumentProvider.eurusd_future(
            expiry_year=2025,
            # Эти параметры могут не подходить для реальных данных Bybit (BTC/USDT spot или perpetual)
            expiry_month=3,
            venue_name=VENUE_NAME,
        )
        logger.success(f"🔧 Инструмент создан: {_INSTRUMENT.id}")

        # Define bar type (Убедитесь, что тип соответствует данным)
        # Жестко задан '1-MINUTE'. Нужно сделать динамически?
        # Если settings.TIMEFRAME всегда '1m', то можно оставить.
        # Иначе нужно конвертировать settings.TIMEFRAME в формат Nautilus (например, '1m' -> '1-MINUTE')
        logger.debug("📊 Создание типа бара...")
        _1MIN_BARTYPE = BarType.from_str(f"{_INSTRUMENT.id}-1-MINUTE-LAST-EXTERNAL")
        # Проверьте, соответствует ли timeframe в _1MIN_BARTYPE тому, который вы загружали
        # Если settings.TIMEFRAME != '1m', это будет ошибка.
        logger.success(f"📊 Тип бара создан: {_1MIN_BARTYPE}")

        # Convert DataFrame rows into Bar objects
        logger.info("🔄 Конвертация DataFrame в объекты Bar...")
        wrangler = BarDataWrangler(_1MIN_BARTYPE, _INSTRUMENT)
        bars_list: list[Bar] = wrangler.process(df)
        logger.success(f"🔄 Конвертация завершена. Всего баров: {len(bars_list)}")

        # Collect and return all prepared data
        prepared_data = {
            "venue_name": VENUE_NAME,
            "instrument": _INSTRUMENT,
            "bar_type": _1MIN_BARTYPE,
            "bars_list": bars_list,
            "data": df  # Добавим сам DataFrame для удобства проверки
        }

        logger.success("🎉 Подготовка данных успешно завершена")
        return prepared_data

    except Exception as e:
        logger.error(f"🔥 Ошибка при подготовке данных из ArcticDB: {e}")
        raise  # Или обработайте ошибку по-другому


# Пример использования (если файл запускается напрямую)
if __name__ == "__main__":
    try:
        logger.info("🏁 Запуск примера использования data_provider")
        data_dict = prepare_data_1min()
        df_data = data_dict.get('data')
        if df_data is not None and not df_data.empty:
            logger.success("✅ Данные успешно загружены из ArcticDB и подготовлены.")
            logger.info(f"📊 Размер данных: {df_data.shape} (строк: {df_data.shape[0]}, колонок: {df_data.shape[1]})")
            logger.info(f"📌 Диапазон данных: с {df_data.index.min()} по {df_data.index.max()}")
            logger.info(f"📊     Превью первых 10 записей:\n{tabulate(df_data.head(10), headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False)}")
            logger.info(f"📊     Превью последних 10 записей:\n{tabulate(df_data.tail(10), headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False)}")
        else:
            logger.warning("⚠️ Данные пусты или не были загружены.")

    except Exception as e:
        logger.error(f"🔥 Ошибка при запуске скрипта: {e}")
        import traceback

        logger.debug(traceback.format_exc())  # Для более подробного вывода ошибки
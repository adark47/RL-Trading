# data/get_data_to_arcticDB.py

import requests
import pandas as pd
import datetime
import time
import sys
from loguru import logger
from tabulate import tabulate
import arcticdb as adb

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
    f"{log_dir}/get_data_to_arcticDB_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


def timeframe_to_interval(timeframe: str) -> str:
    """Преобразует человеко-читаемый таймфрейм в формат, понятный Bybit API."""
    timeframe = timeframe.lower()
    mapping = {
        '1m': '1',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '2h': '120',
        '4h': '240',
        '6h': '360',
        '12h': '720',
        '1d': 'D',
        '1w': 'W',
        '1mth': 'M'
    }

    if timeframe not in mapping:
        logger.warning(f"⚠️ Неизвестный таймфрейм: {timeframe}. Используем значение по умолчанию: 1m")
        return '1'

    return mapping[timeframe]


def validate_parameters(category: str, symbol: str, timeframe: str, days: float):
    """Проверяет корректность входных параметров."""
    valid_categories = ['spot', 'linear', 'inverse']
    if category not in valid_categories:
        raise ValueError(f"Некорректный тип рынка. Допустимые значения: {valid_categories}")

    if not symbol or not isinstance(symbol, str):
        raise ValueError("Символ должен быть непустой строкой")

    if days <= 0:
        raise ValueError("Период должен быть положительным числом")

    # Проверка таймфрейма происходит внутри timeframe_to_interval


def fetch_klines(category: str, symbol: str, timeframe: str = '1m', days: float = 1.0):
    """
    Получает исторические данные свечей с Bybit.

    Параметры:
    - category: 'spot' для спота, 'linear' для USDT фьючерсов, 'inverse' для BTC фьючерсов
    - symbol: например, 'BTCUSDT' для спота или 'BTCUSDT.P' для перпетуал фьючерсов
    - timeframe: таймфрейм свечей (по умолчанию '1m')
    - days: период в днях (может быть дробным, например 0.5 для 12 часов)

    Возвращает DataFrame с колонками: date, open, high, low, close, volume
    """
    # Валидация параметров
    validate_parameters(category, symbol, timeframe, days)

    interval_str = timeframe_to_interval(timeframe)
    logger.info(f"🔍 Запрашиваем данные: {symbol} ({category}), {timeframe}, {days} дней")

    # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: точный расчет временного диапазона ===
    end_datetime = datetime.datetime.now()
    start_datetime = end_datetime - datetime.timedelta(days=days)

    logger.debug(f"Запрашиваем данные за период: с {start_datetime} по {end_datetime} ({days} дней)")

    # === ИСПРАВЛЕНИЕ: Bybit имеет ограничение на исторические данные ===
    MAX_PERIOD_DAYS = 50  # Максимальный период за один запрос
    current_start = start_datetime
    all_data = []
    base_max_period = 50

    # Разбиваем общий период на подпериоды
    while current_start < end_datetime:
        # Определяем конец текущего подпериода
        period_end = min(current_start + datetime.timedelta(days=MAX_PERIOD_DAYS), end_datetime)

        logger.info(
            f"🔄 Запрашиваем данные за период: {current_start.strftime('%Y-%m-%d %H:%M')} - {period_end.strftime('%Y-%m-%d %H:%M')}")

        # Конвертируем в миллисекунды для текущего подпериода
        start_time = int(current_start.timestamp() * 1000)
        end_time = int(period_end.timestamp() * 1000)

        # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: УБРАНЫ ВСЕ ЛИШНИЕ ПРОБЕЛЫ В URL ===
        url = "https://api.bybit.com/v5/market/kline"  # Правильный URL без пробелов!

        current_end = end_time
        period_data = []
        max_attempts = 100
        attempts = 0
        period_retries = 0
        max_period_retries = 5

        while start_time < current_end and attempts < max_attempts:
            attempts += 1

            params = {
                'category': category,
                'symbol': symbol,
                'interval': interval_str,
                'start': start_time,
                'end': current_end,
                'limit': 1000
            }

            try:
                logger.debug(f"📡 Отправка запроса: {params}")
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # Проверка успешности запроса
                if data.get('retCode') != 0:
                    error_msg = data.get('retMsg', 'Неизвестная ошибка')
                    logger.error(f"❌ Ошибка API: {error_msg} (код: {data.get('retCode')})")

                    # Проверка на rate limit
                    if data.get('retCode') == 10001 and "rate" in str(data.get('retMsg', '')).lower():
                        logger.warning("⏳ Достигнут лимит запросов, пауза 2 секунды...")
                        time.sleep(2)
                        continue

                    # Проверка на слишком большой период
                    if data.get('retCode') == 100027 or "period" in str(data.get('retMsg', '')).lower():
                        logger.warning(
                            f"⚠️ Запрошенный период слишком велик ({MAX_PERIOD_DAYS} дней), уменьшаем размер подпериода")
                        MAX_PERIOD_DAYS = max(1, MAX_PERIOD_DAYS // 2)
                        period_retries += 1

                        if period_retries > max_period_retries:
                            logger.error(
                                f"❌ Не удалось запросить данные после {max_period_retries} попыток уменьшения периода")
                            break

                        # Сбрасываем внутренний цикл и начинаем с уменьшенным периодом
                        break

                    # Обработка других ошибок
                    break

                # Проверка структуры ответа
                if not isinstance(data, dict) or 'result' not in data or not isinstance(data['result'], dict):
                    logger.error("❌ Некорректная структура ответа API")
                    break

                result = data['result']
                if 'list' not in result or not isinstance(result['list'], list) or len(result['list']) == 0:
                    logger.info("ℹ️ Больше нет данных для загрузки")
                    break

                klines = result['list']

                # Bybit возвращает данные в порядке от новых к старым
                first_ts = int(klines[0][0])  # самая свежая
                last_ts = int(klines[-1][0])  # самая старая

                # Форматируем временные метки для лога
                oldest_str = datetime.datetime.fromtimestamp(last_ts / 1000).strftime("%H:%M %d.%m.%Y")
                newest_str = datetime.datetime.fromtimestamp(first_ts / 1000).strftime("%H:%M %d.%m.%Y")

                period_data.extend(klines)

                # Следующий запрос должен получить данные до временной метки самой старой свечи
                current_end = last_ts - 1

                logger.info(
                    f"📥 Получено {len(klines)} свечей ({oldest_str} - {newest_str}). Всего в периоде: {len(period_data)}")

                # Соблюдаем rate limit
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                logger.error(f"📡 Ошибка сети: {str(e)}")
                logger.info("⏳ Повторная попытка через 1 секунду...")
                time.sleep(1)
            except Exception as e:
                logger.exception(f"🚨 Неожиданная ошибка: {str(e)}")
                break

        # Если период был уменьшен из-за ошибки, проверяем, удалось ли получить данные
        if period_retries > 0 and len(period_data) > 0:
            logger.info(f"✅ Успешно получены данные с уменьшенным периодом ({MAX_PERIOD_DAYS} дней)")
            # После успешного запроса с уменьшенным периодом, возвращаем базовое значение
            MAX_PERIOD_DAYS = base_max_period
            period_retries = 0

        if period_data:
            all_data.extend(period_data)
            logger.success(
                f"✅ Успешно получены данные за период {current_start.strftime('%Y-%m-%d')} - {period_end.strftime('%Y-%m-%d')}")
        else:
            logger.warning(
                f"⚠️ Не удалось получить данные за период {current_start.strftime('%Y-%m-%d')} - {period_end.strftime('%Y-%m-%d')}")

            # Если период был уменьшен до 1 дня и все равно нет данных, пропускаем этот период
            if MAX_PERIOD_DAYS <= 1:
                logger.error("❌ Не удалось получить данные даже с минимальным периодом. Пропускаем этот период.")
                # Возвращаем базовое значение для следующих периодов
                MAX_PERIOD_DAYS = base_max_period
                period_retries = 0

        # Переходим к следующему подпериоду
        current_start = period_end
        time.sleep(0.5)

    if not all_data:
        logger.error("❌ Не удалось получить данные свечей")
        return pd.DataFrame()

    # Преобразуем в DataFrame
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    df = pd.DataFrame(all_data, columns=columns)

    # Оставляем только необходимые колонки
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    # Конвертируем date в читаемый формат
    df['date'] = pd.to_datetime(df['date'].astype(int), unit='ms')

    # Конвертируем числовые колонки
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Заменяем NaN на 0 для объема, для цен оставляем NaN
        if col == 'volume':
            df[col] = df[col].fillna(0)

    # Сортируем по времени (от старых к новым)
    df = df.sort_values('date').reset_index(drop=True)

    # Проверка покрытия запрашиваемого периода
    if not df.empty:
        actual_days = (df['date'].max() - df['date'].min()).total_seconds() / (24 * 60 * 60)
        logger.info(f"📊 Получены данные за {actual_days:.2f} дней из запрошенных {days} дней")
        logger.info(f"📌 Диапазон данных: с {df['date'].min()} по {df['date'].max()}")
    else:
        logger.warning("⚠️ Получен пустой набор данных")

    logger.success(f"✅ Успешно получено {len(df)} свечей для {symbol}")
    return df


def get_existing_data(library, symbol_name):
    """Получает существующие данные из ArcticDB, если они есть"""
    if library.has_symbol(symbol_name):
        logger.info(f"🔍 Найдены существующие данные для {symbol_name}")
        existing_data = library.read(symbol_name).data
        logger.info(
            f"📌 В базе уже есть данные за период: с {existing_data['date'].min()} по {existing_data['date'].max()}")
        logger.info(f"📊 Количество существующих записей: {len(existing_data)}")
        return existing_data
    return None


def save_to_arcticdb(library, symbol_name, new_data, existing_data=None):
    """Сохраняет данные в ArcticDB с поддержкой инкрементальных обновлений"""
    if existing_data is not None and not existing_data.empty:
        # Определяем последнюю дату в существующих данных
        last_date = existing_data['date'].max()
        logger.info(f"⏳ Определяем новые данные начиная с {last_date}")

        # Фильтруем новые данные, оставляя только те, что новее последней записи в базе
        new_data = new_data[new_data['date'] > last_date]

        # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: сброс индекса для совместимости с ArcticDB ===
        if not new_data.empty:
            new_data = new_data.reset_index(drop=True)
            logger.info(f"📥 Добавляем {len(new_data)} новых записей")
            # Добавляем новые данные к существующим
            library.append(symbol_name, new_data)
            logger.success(f"✅ Успешно добавлено {len(new_data)} новых записей в {symbol_name}")
        else:
            logger.info("ℹ️ Новых данных для добавления нет")
    else:
        # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: сброс индекса перед первоначальным сохранением ===
        if not new_data.empty:
            new_data = new_data.reset_index(drop=True)
            # Сохраняем данные как новый символ
            logger.info(f"💾 Сохраняем {len(new_data)} записей как новые данные в {symbol_name}")
            library.write(symbol_name, new_data)
            logger.success(f"✅ Данные успешно сохранены в {symbol_name}")
        else:
            logger.warning("⚠️ Нет данных для сохранения")


if __name__ == "__main__":
    # Параметры запроса
    MARKET_TYPE = 'linear'  # 'spot', 'linear' (USDT фьючерсы) или 'inverse' (BTC фьючерсы)
    SYMBOL = 'DOGEUSDT'  # Для спота или 'BTCUSDT.P' для перпетуал фьючерсов
    TIMEFRAME = '1m'  # Таймфрейм (по умолчанию 1 минута)
    DAYS = 100.0  # Период в днях (можно дробное значение)

    logger.info(f"🚀 Начинаем загрузку исторических данных с Bybit")
    logger.info(f"Параметры: {MARKET_TYPE} рынок, {SYMBOL}, {TIMEFRAME}, {DAYS} дней")
    logger.warning(f"⚠️ Внимание: Bybit API имеет ограничение на исторические данные (~70 дней для минутных свечей). "
                   f"Будет выполнена автоматическая разбивка на подпериоды по 50 дней с возможной адаптацией.")

    try:
        # === ИНИЦИАЛИЗАЦИЯ ARCTICDB ===
        storage_path = "arcticdb_storage"
        logger.info(f"🔧 Инициализация ArcticDB хранилища: {storage_path}")
        ac = adb.Arctic(f"lmdb://{storage_path}")

        # Создаем или получаем библиотеку для хранения данных
        library_name = "bybit_market_data"
        if not ac.has_library(library_name):
            ac.create_library(library_name)
            logger.info(f"🆕 Создана новая библиотека ArcticDB: {library_name}")

        library = ac.get_library(library_name)
        logger.info(f"🗄️ Используем библиотеку ArcticDB: {library_name}")

        # Формируем имя символа для хранения
        symbol_name = f"{SYMBOL}_{TIMEFRAME}_{MARKET_TYPE}"
        logger.info(f"🏷️ Имя символа в ArcticDB: {symbol_name}")

        # Получаем существующие данные, если они есть
        existing_data = get_existing_data(library, symbol_name)

        # Определяем, сколько данных нужно запросить
        if existing_data is not None and not existing_data.empty:
            # Рассчитываем, сколько дней прошло с последней записи до текущего времени
            last_date = existing_data['date'].max()
            time_diff = datetime.datetime.now() - last_date
            days_to_fetch = max(0.1, time_diff.total_seconds() / (24 * 3600))  # минимум 0.1 дня (2.4 часа)
            logger.info(f"🕒 Последние данные в базе: {last_date}")
            logger.info(f"⏳ Запрашиваем данные за последний {days_to_fetch:.2f} дней")
        else:
            days_to_fetch = DAYS
            logger.info(f"📥 Запрашиваем данные за {days_to_fetch} дней (полная загрузка)")

        # Получаем данные
        df = fetch_klines(MARKET_TYPE, SYMBOL, TIMEFRAME, days_to_fetch)

        # Обрабатываем и сохраняем данные
        if not df.empty:
            # Сохраняем данные в ArcticDB с поддержкой инкрементальных обновлений
            save_to_arcticdb(library, symbol_name, df, existing_data)

            # Проверяем обновленные данные
            updated_data = get_existing_data(library, symbol_name)
            if updated_data is not None and not updated_data.empty:
                logger.info(f"📊 Общее количество записей после обновления: {len(updated_data)}")
                logger.info(
                    f"📌 Обновленный диапазон данных: с {updated_data['date'].min()} по {updated_data['date'].max()}")

                # Выводим превью данных
                logger.info("📊 Превью данных:")
                logger.info(f"\n {tabulate(updated_data.head(5), showindex=True, headers='keys', tablefmt='psql')}")
                logger.info(f"\n {tabulate(updated_data.tail(5), showindex=True, headers='keys', tablefmt='psql')}")
        else:
            logger.warning("⚠️ Нет данных для сохранения")

    except Exception as e:
        logger.exception(f"🔥 Критическая ошибка в основном потоке: {str(e)}")
        sys.exit(1)
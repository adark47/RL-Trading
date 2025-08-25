import os
import time
import pandas as pd
from datetime import datetime
from binance.client import Client
from loguru import logger

# Настройка логгера
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Удаляем стандартный обработчик и добавляем свои
logger.remove()
logger.add(
    f"{log_dir}/binance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    serialize=False
)
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


def setup_binance_client():
    """Инициализация клиента Binance с обработкой ошибок"""
    try:
        logger.info("Инициализация Binance API клиента...")
        client = Client()
        # Проверка соединения
        client.get_server_time()
        logger.success("Успешное подключение к Binance API")
        return client
    except Exception as e:
        logger.critical(f"Критическая ошибка при подключении к Binance API: {str(e)}")
        raise ConnectionError(
            "Не удалось подключиться к Binance API. Проверьте интернет-соединение и статус API.") from e


def get_historical_data(symbol, interval, start_str, end_str=None, limit=1000):
    """
    Получает исторические данные свечей с Binance с поддержкой пагинации и детальным логированием

    Параметры:
    symbol - торговая пара (например, 'BTCUSDT')
    interval - интервал свечей (Client.KLINE_INTERVAL_1MINUTE и т.д.)
    start_str - начало периода (формат 'YYYY-MM-DD' или timestamp)
    end_str - конец периода (необязательно)
    limit - максимальное количество свечей за один запрос (макс. 1000)

    Возвращает DataFrame с колонками:
    timestamp, open, high, low, close, volume, volume_weighted_average, num_trades
    """
    client = setup_binance_client()
    all_klines = []

    try:
        # Конвертация дат в timestamp
        start_ts = int(time.mktime(time.strptime(start_str, "%Y-%m-%d"))) * 1000
        end_ts = None
        if end_str:
            end_ts = int(time.mktime(time.strptime(end_str, "%Y-%m-%d"))) * 1000

        logger.info(f"Запрос данных для {symbol} с {start_str} по {end_str or 'текущее время'}")
        logger.debug(f"Интервал: {interval}, Лимит: {limit} свечей на запрос")
        logger.debug(f"Внутренние временные метки: start={start_ts}, end={end_ts}")

        initial_start_ts = start_ts
        request_count = 0
        total_candles = 0

        while True:
            request_count += 1
            logger.debug(f"Запрос #{request_count} к API: startTime={datetime.fromtimestamp(start_ts / 1000)}")

            try:
                klines = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=start_ts,
                    endTime=end_ts
                )
            except Exception as e:
                logger.error(f"Ошибка при запросе данных от {datetime.fromtimestamp(start_ts / 1000)}: {str(e)}")
                if "Invalid interval" in str(e):
                    logger.critical(
                        f"Некорректный интервал: {interval}. Допустимые значения: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M")
                raise

            if not klines:
                logger.info("Больше нет данных для загрузки")
                break

            candles_count = len(klines)
            total_candles += candles_count
            all_klines.extend(klines)

            # Логирование прогресса
            if request_count % 5 == 0 or candles_count < limit:
                start_time = datetime.fromtimestamp(klines[0][0] / 1000).strftime("%Y-%m-%d %H:%M")
                end_time = datetime.fromtimestamp(klines[-1][0] / 1000).strftime("%Y-%m-%d %H:%M")
                logger.info(f"Загружено {candles_count} свечей | Общее количество: {total_candles} | "
                            f"Диапазон: {start_time} - {end_time}")

            # Если получили меньше лимита, значит это последние данные
            if candles_count < limit:
                break

            # Устанавливаем новую стартовую точку (последняя свеча + 1)
            start_ts = klines[-1][0] + 1

            # Добавляем небольшую задержку для соблюдения лимитов API
            time.sleep(0.1)

        logger.success(f"Завершено! Всего получено {total_candles} свечей за {request_count} запросов")

        # Подготовка данных для DataFrame
        data = []
        for k in all_klines:
            # Расчет volume_weighted_average = quote asset volume / volume
            volume = float(k[5])
            quote_volume = float(k[7])
            vwap = quote_volume / volume if volume != 0 else 0

            data.append([
                pd.to_datetime(k[0], unit='ms'),  # date
                float(k[1]),  # open
                float(k[2]),  # high
                float(k[3]),  # low
                float(k[4]),  # close
                volume,  # volume
                vwap,  # volume_weighted_average
                int(k[8])  # num_trades
            ])

        # Создание DataFrame
        df = pd.DataFrame(data, columns=[
            'date', 'open', 'high', 'low', 'close',
            'volume', 'volume_weighted_average', 'num_trades'
        ])

        # Дополнительная статистика
        duration = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        logger.info(f"Статистика данных: {len(df)} свечей за {duration} дней")
        logger.debug(f"Первая свеча: {df['date'].iloc[0]} | Последняя свеча: {df['date'].iloc[-1]}")

        return df

    except ValueError as ve:
        logger.error(f"Ошибка формата даты: {str(ve)}")
        logger.debug("Пример корректного формата: '2023-01-01'")
        raise
    except Exception as e:
        logger.exception(f"Необработанная ошибка при получении исторических данных: {str(e)}")
        raise


def save_to_csv(df, symbol, interval, start_date, end_date):
    """Сохраняет DataFrame в CSV с проверкой директории"""
    try:
        data_dir = "./"
        os.makedirs(data_dir, exist_ok=True)

        # Форматирование имени файла
        interval_str = interval.replace("Client.KLINE_INTERVAL_", "").lower()
        filename = f"data.csv"
        filepath = os.path.join( filename)

        # Сохранение данных
        df.to_csv(filepath, index=False)

        # Логирование результата
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # в МБ
        logger.success(f"Данные успешно сохранены в {filepath}")
        logger.info(f"Размер файла: {file_size:.2f} MB | Строк: {len(df)}")

        return filepath
    except Exception as e:
        logger.error(f"Ошибка при сохранении CSV: {str(e)}")
        raise


def main():
    """Основная функция с обработкой конфигурации"""
    symbol = "DOGEUSDT"
    interval = Client.KLINE_INTERVAL_1MINUTE
    start_date = "2025-01-01"
    end_date = "2025-08-17"  # Для 1-минутных данных лучше брать небольшие периоды

    logger.info("=" * 50)
    logger.info(f"ЗАПУСК СБОРЩИКА ДАННЫХ Binance | {symbol} | {interval}")
    logger.info(f"Период: {start_date} по {end_date}")
    logger.info("=" * 50)

    try:
        # Получение данных
        start_time = time.time()
        df = get_historical_data(symbol, interval, start_date, end_date)
        processing_time = time.time() - start_time

        # Сохранение в CSV
        filepath = save_to_csv(df, symbol, interval, start_date, end_date)

        # Итоговая статистика
        logger.success(f"ПРОЦЕСС ЗАВЕРШЕН УСПЕШНО!")
        logger.info(f"Время выполнения: {processing_time:.2f} секунд")
        logger.info(f"Средняя скорость: {len(df) / processing_time:.2f} свечей/сек")

        # Пример данных
        if not df.empty:
            logger.debug("Пример первых 3 строк данных:")
            for i in range(min(3, len(df))):
                logger.debug(f"{df.iloc[i]['date']} | "
                             f"O:{df.iloc[i]['open']:.2f} H:{df.iloc[i]['high']:.2f} "
                             f"L:{df.iloc[i]['low']:.2f} C:{df.iloc[i]['close']:.2f} "
                             f"Vol:{df.iloc[i]['volume']:.2f}")

        return filepath

    except Exception as e:
        logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА В ПРОЦЕССЕ СБОРКИ ДАННЫХ: {str(e)}")
        logger.debug("Рекомендуемые действия:")
        logger.debug("1. Проверьте корректность торговой пары")
        logger.debug("2. Убедитесь, что даты в правильном формате")
        logger.debug("3. Проверьте подключение к интернету")
        logger.debug("4. Убедитесь, что интервал поддерживается Binance")
        return None


if __name__ == "__main__":
    # Проверка установки необходимых пакетов
    try:
        import binance

        logger.debug(f"Используется python-binance версии: {binance.__version__}")
    except ImportError:
        logger.critical("Требуемый пакет 'python-binance' не установлен!")
        logger.info("Установите его командой: pip install python-binance")
        exit(1)

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Процесс прерван пользователем (Ctrl+C)")
        logger.info("Работа программы завершена штатно")
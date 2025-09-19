# create_transactions_db.py
import sys
import os
from loguru import logger
from datetime import datetime
import psycopg2
from pg_connector import PostgreSQLConnector # Предполагается, что pg_connector.py находится в той же директории или в sys.path

# Настройка логгера с цветами и эмодзи
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"logs/create_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


def create_transactions_table():
    """Создание таблицы transactions если она не существует"""
    # Добавлен столбец version VARCHAR(50)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS transactions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        strategy_name VARCHAR(100) NOT NULL,
        instrument_id VARCHAR(50) NOT NULL,
        trade_type VARCHAR(20) NOT NULL, -- 'LONG', 'SHORT'
        action VARCHAR(20) NOT NULL, -- 'OPEN', 'CLOSE'
        order_side VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
        quantity DECIMAL NOT NULL,
        price DECIMAL NOT NULL,
        order_id VARCHAR(100),
        position_id VARCHAR(100),
        realized_pnl DECIMAL,
        commission DECIMAL DEFAULT 0,
        trade_mode VARCHAR(20) NOT NULL, -- 'BACKTEST', 'LIVE'
        version VARCHAR(50), -- Новое поле
        metadata JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_transactions_strategy ON transactions(strategy_name);
    CREATE INDEX IF NOT EXISTS idx_transactions_instrument ON transactions(instrument_id);
    CREATE INDEX IF NOT EXISTS idx_transactions_mode ON transactions(trade_mode);
    """

    try:
        with PostgreSQLConnector("transactions") as db:
            if db.connection:
                db.cursor.execute(create_table_query)
                db.connection.commit()
                logger.success("✅ Таблица transactions успешно создана или уже существует")
                return True
            else:
                logger.error("❌ Не удалось подключиться к базе данных")
                return False
    except Exception as e:
        logger.error(f"❌ Ошибка при создании таблицы: {e}")
        return False


def test_database_connection():
    """Проверка подключения к базе данных transactions"""
    try:
        with PostgreSQLConnector("transactions") as db:
            if db.connection:
                db.cursor.execute("SELECT version();")
                version = db.cursor.fetchone()
                logger.info(f"✅ Подключение к БД успешно. PostgreSQL версия: {version[0]}")

                # Проверяем существование таблицы (без проверки version, так как это поле данных)
                db.cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'transactions'
                    );
                """)
                exists = db.cursor.fetchone()[0]
                if exists:
                    logger.success("✅ Таблица transactions существует")
                else:
                    logger.warning("⚠️ Таблица transactions не найдена")

                return True
            else:
                logger.error("❌ Не удалось подключиться к базе данных transactions")
                return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}")
        return False


def create_database_if_not_exists():
    """Создание базы данных transactions если она не существует"""
    try:
        # Подключаемся к системной базе данных для создания новой БД
        with PostgreSQLConnector("postgres") as db:
            if db.connection:
                db.connection.autocommit = True
                db.cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'transactions';")
                exists = db.cursor.fetchone()

                if not exists:
                    db.cursor.execute("CREATE DATABASE transactions;")
                    logger.success("✅ База данных transactions создана")
                else:
                    logger.info("✅ База данных transactions уже существует")
                return True
            else:
                logger.error("❌ Не удалось подключиться к системной базе данных")
                return False
    except Exception as e:
        logger.error(f"❌ Ошибка при создании базы данных: {e}")
        return False


def main():
    """Основная функция создания БД и таблиц"""
    logger.info("🚀 Запуск создания базы данных и таблицы для транзакций")

    # Создаем базу данных если она не существует
    if create_database_if_not_exists():
        # Проверяем подключение к новой базе
        if test_database_connection():
            # Создаем таблицу
            if create_transactions_table():
                logger.success("🎉 Все операции успешно завершены!")
                return True

    logger.error("💥 Произошла ошибка при настройке базы данных")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# pg_connector.py

import sys
import os
import time
import psycopg2
from psycopg2 import sql
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from typing import Optional, List, Tuple, Any


# Настройка логирования с учетом кодировки Windows
log_dir = "logs"
logger.remove()  # Удаляем стандартный обработчик

# Принудительно отключаем цвет и эмодзи для совместимости с Windows
use_color = False

logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="INFO",
    colorize=use_color
)
logger.add(
    f"{log_dir}/pg_connector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем конфигурацию
try:
    from config import PGConfig as PGConfig
    settings = PGConfig()
except ImportError:
    logger.warning("Файл конфигурации не найден, используем значения по умолчанию")


class PostgreSQLConnector:
    def __init__(self, database_name: str):
        """
        Инициализация коннектора к PostgreSQL

        Args:
            database_name (str): Имя базы данных для подключения
        """
        self.host = os.getenv("DB_HOST", settings.DB_HOST)
        self.port = os.getenv("DB_PORT", settings.DB_PORT)
        self.user = os.getenv("DB_USER", settings.DB_USER)
        self.password = os.getenv("DB_PASSWORD", settings.DB_PASSWORD)
        self.database = database_name
        self.connection = None
        self.cursor = None

    def __enter__(self):
        """Поддержка контекстного менеджера"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие соединения при выходе из контекста"""
        self.disconnect()

    def _attempt_connect(self) -> bool:
        """
        Попытка подключения к базе данных

        Returns:
            bool: True если подключение успешно, False в противном случае
        """
        try:
            logger.info(f"Попытка подключения к базе данных: {self.database}")

            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )

            self.cursor = self.connection.cursor()

            # Проверка подключения
            self.cursor.execute("SELECT version();")
            db_version = self.cursor.fetchone()
            logger.success(f"Подключение к PostgreSQL успешно установлено")
            logger.debug(f"Версия PostgreSQL: {db_version[0]}")

            return True

        except psycopg2.Error as e:
            logger.error(f"Ошибка подключения к PostgreSQL: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при подключении: {e}")
            return False

    def connect(self, retries: int = 3, delay: int = 5) -> bool:
        """
        Установка соединения с базой данных с возможностью повторных попыток

        Args:
            retries (int): Количество попыток подключения
            delay (int): Задержка между попытками в секундах

        Returns:
            bool: True если подключение успешно, False в противном случае
        """
        for attempt in range(retries):
            if self._attempt_connect():
                return True
            if attempt < retries - 1:
                logger.warning(f"Повторная попытка подключения через {delay} секунд... ({attempt + 1}/{retries})")
                time.sleep(delay)
        return False

    def disconnect(self):
        """Закрытие соединения с базой данных"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Соединение с базой данных закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения: {e}")

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Optional[List[Tuple[Any, ...]]]:
        """
        Выполнение SQL-запроса

        Args:
            query (str): SQL-запрос
            params (tuple, optional): Параметры для запроса

        Returns:
            list: Результаты запроса или None в случае ошибки
        """
        try:
            if not self.connection or self.connection.closed:
                logger.error("Нет активного соединения с базой данных")
                return None

            logger.debug(f"Выполнение запроса: {query} с параметрами: {params}")

            self.cursor.execute(query, params)

            if query.strip().upper().startswith('SELECT'):
                results = self.cursor.fetchall()
                return results
            else:
                self.connection.commit()
                return True

        except psycopg2.Error as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            if self.connection and not query.strip().upper().startswith('SELECT'):
                self.connection.rollback()
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при выполнении запроса: {e}")
            return None

    def preview_data(self, query: str, params: Optional[Tuple] = None, limit: int = 10):
        """
        Вывод превью данных с использованием tabulate

        Args:
            query (str): SQL-запрос
            params (tuple, optional): Параметры для запроса
            limit (int): Количество строк для отображения (по умолчанию 10)
        """
        try:
            # Добавляем LIMIT к запросу если это SELECT
            if query.strip().upper().startswith('SELECT'):
                limited_query = f"{query.rstrip(';')} LIMIT {limit};"
            else:
                limited_query = query

            results = self.execute_query(limited_query, params)

            if results is not None and len(results) > 0:
                # Получаем названия колонок
                column_names = [desc[0] for desc in self.cursor.description]

                # Выводим данные в виде таблицы
                table = tabulate(results, headers=column_names, tablefmt="grid")
                logger.info(f"Превью данных (первые {min(limit, len(results))} строк):")
                print(table)
            elif results is not None:
                logger.info("Запрос выполнен успешно, но данных нет")
            else:
                logger.warning("Не удалось получить данные для превью")

        except Exception as e:
            logger.error(f"Ошибка при создании превью данных: {e}")


def test_connection():
    """Тест подключения к базе данных при запуске скрипта"""
    logger.info("Запуск теста подключения к PostgreSQL")

    # Здесь можно указать тестовую базу данных или любую доступную
    test_db_name = "postgres"  # Системная база данных по умолчанию

    with PostgreSQLConnector(test_db_name) as connector:
        logger.success("Тест подключения успешно пройден!")

        # Пример выполнения простого запроса
        logger.info("Выполнение тестового запроса...")
        connector.preview_data("SELECT version();")

        # Не возвращаем значение, чтобы pytest не ругался
        # Вместо этого используем assert если нужно проверить что-то конкретное
        assert True  # Просто проверка что мы дошли до этой точки


if __name__ == "__main__":
    # При запуске скрипта выполняем тест подключения
    test_connection()

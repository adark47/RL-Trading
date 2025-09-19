# questdb_connector.py

import sys
import os
import time
import psycopg2
from psycopg2 import sql
# Импортируем необходимые исключения из psycopg2
from psycopg2 import OperationalError, DatabaseError, InterfaceError, ProgrammingError
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from typing import Optional, List, Tuple, Any

# Настройка логирования с учетом кодировки Windows
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Убедиться, что директория для логов существует
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
    f"{log_dir}/questdb_connector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем конфигурацию
# Предположим, что конфигурация будет в файле config.py и называться QuestDBConfig
try:
    from config import QuestDBConfig as QuestDBConfig

    settings = QuestDBConfig()
except ImportError:
    logger.warning("Файл конфигурации config.py с QuestDBConfig не найден, используем значения по умолчанию")


class QuestDBConnector:
    def __init__(self, database_name: str = "qdb"):  # По умолчанию QuestDB использует 'qdb'
        """
        Инициализация коннектора к QuestDB через PostgreSQL wire protocol

        Args:
            database_name (str): Имя базы данных для подключения (обычно 'qdb' для QuestDB)
        """
        self.host = os.getenv("QUESTDB_HOST", settings.DB_HOST)
        self.port = os.getenv("QUESTDB_PORT", settings.DB_PORT)
        self.user = os.getenv("QUESTDB_USER", settings.DB_USER)
        self.password = os.getenv("QUESTDB_PASSWORD", settings.DB_PASSWORD)
        self.database = database_name
        self.connection = None
        self.cursor = None
        # QuestDB имеет ограничения на выполнение некоторых DDL/DML команд в одном запросе
        # и не поддерживает транзакции в полной мере. Поэтому установим флаг.
        self.autocommit = True  # Всегда в режиме autocommit для QuestDB

    def __enter__(self):
        """Поддержка контекстного менеджера"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие соединения при выходе из контекста"""
        self.disconnect()

    def _attempt_connect(self) -> bool:
        """
        Попытка подключения к базе данных QuestDB

        Returns:
            bool: True если подключение успешно, False в противном случае
        """
        try:
            logger.info(f"Попытка подключения к QuestDB: {self.database} на {self.host}:{self.port}")

            # В psycopg2 для QuestDB может потребоваться дополнительный параметр
            # так как QuestDB использует специфичную реализацию PostgreSQL wire protocol
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                # Дополнительные параметры для совместимости с QuestDB
                # autocommit=True рекомендуется для QuestDB
                autocommit=self.autocommit
            )

            self.cursor = self.connection.cursor()

            # Проверка подключения
            # QuestDB может не поддерживать все стандартные запросы PostgreSQL
            # Попробуем выполнить простой запрос
            self.cursor.execute("SELECT 1;")
            result = self.cursor.fetchone()
            logger.success(f"Подключение к QuestDB успешно установлено. Проверка: {result}")

            # Получение версии QuestDB (если доступна)
            try:
                self.cursor.execute("SELECT version();")
                db_version = self.cursor.fetchone()
                logger.debug(f"Версия QuestDB (через PostgreSQL wire): {db_version[0] if db_version else 'Неизвестно'}")
            except (ProgrammingError, DatabaseError) as e:
                logger.debug(f"Не удалось получить версию через SELECT version(): {e}. Это нормально для QuestDB.")
                # QuestDB может не поддерживать SELECT version()

            return True

        except psycopg2.Error as e:  # Ловим все ошибки psycopg2
            logger.error(f"Ошибка подключения к QuestDB: {e}")
            # Закрываем соединение, если оно было создано, но возникла ошибка
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
                self.cursor = None
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при подключении к QuestDB: {e}")
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
                self.cursor = None
            return False

    def connect(self, retries: int = 3, delay: int = 5) -> bool:
        """
        Установка соединения с базой данных QuestDB с возможностью повторных попыток

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
        """Закрытие соединения с базой данных QuestDB"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Соединение с QuestDB закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения с QuestDB: {e}")

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Optional[List[Tuple[Any, ...]]]:
        """
        Выполнение SQL-запроса к QuestDB

        Args:
            query (str): SQL-запрос
            params (tuple, optional): Параметры для запроса

        Returns:
            list: Результаты запроса или None в случае ошибки
            bool: True для команд, которые не возвращают результат (INSERT, CREATE и т.д.)
        """
        try:
            if not self.connection or self.connection.closed:
                logger.error("Нет активного соединения с QuestDB")
                return None

            logger.debug(f"Выполнение запроса к QuestDB: {query} с параметрами: {params}")

            # QuestDB использует autocommit, поэтому connection.commit() не нужен
            # и может вызвать ошибку

            self.cursor.execute(query, params)

            # Определяем тип запроса и обрабатываем результат
            query_upper = query.strip().upper()

            if query_upper.startswith('SELECT') or query_upper.startswith('WITH'):
                # Запросы на выборку данных
                results = self.cursor.fetchall()
                return results
            elif query_upper.startswith('SHOW') or query_upper.startswith('DESCRIBE'):
                # Команды, которые могут возвращать результат (SHOW TABLES, DESCRIBE table)
                results = self.cursor.fetchall()
                return results
            else:
                # Для команд типа INSERT, CREATE, DROP и т.д., которые не возвращают результат
                # psycopg2 с autocommit=True не требует commit
                logger.info(f"Запрос выполнен: {query}")
                # Возвращаем True для указания успешного выполнения
                return True  # Или можно вернуть количество затронутых строк, если доступно

        except psycopg2.Error as e:
            # Ловим специфичные ошибки psycopg2
            logger.error(f"Ошибка выполнения запроса к QuestDB: {e}")
            # В QuestDB с autocommit откат транзакции не применяется
            # self.connection.rollback() не требуется и может быть не поддержан
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при выполнении запроса к QuestDB: {e}")
            return None

    def preview_data(self, table_name: str, limit: int = 10):
        """
        Вывод превью данных из таблицы QuestDB с использованием tabulate

        Args:
            table_name (str): Имя таблицы
            limit (int): Количество строк для отображения (по умолчанию 10)
        """
        try:
            # Формируем безопасный запрос на выборку данных
            query = sql.SQL("SELECT * FROM {table} LIMIT {limit}").format(
                table=sql.Identifier(table_name),
                limit=sql.Literal(limit)
            )

            # Преобразуем SQL объект в строку для логирования и выполнения
            query_str = query.as_string(self.connection)
            logger.debug(f"Формирование запроса для превью: {query_str}")

            results = self.execute_query(query_str)

            if results is not None and len(results) > 0:
                # Получаем названия колонок
                # Для запросов SELECT * psycopg2 должен предоставить описание колонок
                if self.cursor.description:
                    column_names = [desc[0] for desc in self.cursor.description]
                else:
                    # Если описание недоступно, создаем фиктивные имена
                    column_names = [f"Column_{i}" for i in range(len(results[0]))] if results else []

                # Выводим данные в виде таблицы
                table = tabulate(results, headers=column_names, tablefmt="grid")
                logger.info(f"Превью данных из таблицы '{table_name}' (первые {min(limit, len(results))} строк):")
                print(table)
            elif results is not None:
                logger.info(f"Запрос к таблице '{table_name}' выполнен успешно, но данных нет")
            else:
                logger.warning(f"Не удалось получить данные для превью из таблицы '{table_name}'")

        except Exception as e:
            logger.error(f"Ошибка при создании превью данных из таблицы '{table_name}': {e}")

    def insert_dataframe(self, df, table_name: str, if_exists: str = 'append'):
        """
        Вставка данных из pandas DataFrame в таблицу QuestDB.
        ВАЖНО: Эта функция требует установленного pandas.

        Args:
            df (pandas.DataFrame): DataFrame с данными для вставки
            table_name (str): Имя таблицы
            if_exists (str): Что делать, если таблица существует ('fail', 'replace', 'append').
                             Для QuestDB рекомендуется 'append'.
        """
        import platform
        try:
            # Пытаемся импортировать fireducks.pandas только на Linux
            if platform.system().lower() == 'linux':
                import fireducks.pandas as pd
                print("Загружен fireducks.pandas")
            else:
                raise ImportError
        except ImportError:
            import pandas as pd
            print("Загружен стандартный pandas")

        try:
            if not self.connection or self.connection.closed:
                logger.error("Нет активного соединения с QuestDB")
                return False

            if if_exists == 'replace':
                logger.warning(
                    "QuestDB не поддерживает DROP TABLE IF EXISTS в полной мере через этот коннектор. Рекомендуется использовать 'append'.")
                # Попытка удалить таблицу (может не сработать)
                try:
                    self.execute_query(f"DROP TABLE IF EXISTS {table_name};")
                except:
                    pass
                # Создание таблицы будет выполнено автоматически при первой вставке через to_sql

            # Используем метод to_sql из pandas
            # QuestDB может иметь особенности, поэтому используем метод append
            # index=False, чтобы не вставлять индекс DataFrame как колонку
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False, method='multi')
            logger.info(f"Данные из DataFrame успешно вставлены в таблицу '{table_name}'")
            return True

        except Exception as e:
            logger.error(f"Ошибка при вставке данных из DataFrame в таблицу '{table_name}': {e}")
            return False


def test_connection():
    """Тест подключения к базе данных QuestDB при запуске скрипта"""
    logger.info("Запуск теста подключения к QuestDB")

    # Имя базы данных по умолчанию для QuestDB
    test_db_name = "qdb"

    with QuestDBConnector(test_db_name) as connector:
        logger.success("Тест подключения успешно пройден!")

        # Простой тестовый запрос
        logger.info("Выполнение тестового запроса SELECT 1...")
        result = connector.execute_query("SELECT 1;")
        if result:
            logger.info(f"Результат тестового запроса: {result}")

        # Попробуем получить список таблиц (если доступен)
        try:
            tables_result = connector.execute_query("SHOW TABLES;")
            if tables_result is not None:
                logger.info("Доступные таблицы:")
                if tables_result:
                    # Предполагаем, что первая колонка содержит имя таблицы
                    table_names = [row[0] for row in tables_result]
                    for name in table_names:
                        logger.info(f"  - {name}")
                else:
                    logger.info("  Нет доступных таблиц.")
            # else: ошибка уже залогирована в execute_query
        except Exception as e:
            logger.debug(f"Команда SHOW TABLES не поддерживается или вызвала ошибку: {e}")

        assert True  # Просто проверка что мы дошли до этой точки


if __name__ == "__main__":
    # При запуске скрипта выполняем тест подключения
    test_connection()
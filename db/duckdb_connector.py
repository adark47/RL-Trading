import sys
import os
# Импортируем psycopg2 вместо duckdb
import psycopg2
# from psycopg2.extras import RealDictCursor # Опционально, для получения результатов как словарей
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from typing import Optional, List, Tuple, Any, Union

# --- Настройка логирования (без изменений) ---
log_dir = "logs"
logger.remove()

use_color = False

logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="INFO",
    colorize=use_color
)
logger.add(
    f"{log_dir}/duckdb_connector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Конфигурация (адаптируйте под свои нужды) ---
# Предположим, ваш config.py содержит что-то вроде:
# class DuckDBConfig:
#     DUCKDB_HOST = "192.168.88.6"
#     DUCKDB_PORT = 54321
#     DUCKDB_DATABASE = "my_database" # Или оставьте пустым/None если не требуется
#     DUCKDB_USER = "duckdb_user" # Или оставьте пустым/None если не требуется
#     DUCKDB_PASSWORD = "duckdb_password" # Или оставьте пустым/None если не требуется

try:
    from config import DuckDBConfig as DuckDBConfig
    settings = DuckDBConfig()
except ImportError:
    logger.warning("Файл конфигурации не найден, используем значения по умолчанию")
    # Создаём фиктивный объект настроек или используем значения напрямую
    class DuckDBConfig:
        DUCKDB_HOST = "192.168.88.6"
        DUCKDB_PORT = 54321
        DUCKDB_DATABASE = None      # Может быть необязательным
        DUCKDB_USER = None          # Может быть необязательным
        DUCKDB_PASSWORD = None      # Может быть необязательным
    settings = DuckDBConfig()


class DuckDBConnector:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 database: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Инициализация коннектора к DuckDB через PostgreSQL-совместимый интерфейс.

        Args:
            host (str): Хост сервера DuckDB. По умолчанию из config.
            port (int): Порт сервера DuckDB. По умолчанию из config.
            database (str, optional): Имя базы данных. По умолчанию из config.
            user (str, optional): Имя пользователя. По умолчанию из config.
            password (str, optional): Пароль пользователя. По умолчанию из config.
        """
        self.host = host or getattr(settings, 'DUCKDB_HOST', 'localhost')
        self.port = port or getattr(settings, 'DUCKDB_PORT', 5432)
        self.database = database or getattr(settings, 'DUCKDB_DATABASE', None)
        self.user = user or getattr(settings, 'DUCKDB_USER', None)
        self.password = password or getattr(settings, 'DUCKDB_PASSWORD', None)
        self.connection: Optional[psycopg2.extensions.connection] = None

    def __enter__(self):
        """Поддержка контекстного менеджера"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие соединения при выходе из контекста"""
        self.disconnect()

    def _attempt_connect(self) -> bool:
        """
        Попытка подключения к базе данных через psycopg2.

        Returns:
            bool: True если подключение успешно, False в противном случае.
        """
        try:
            logger.info(f"Попытка подключения к DuckDB (PostgreSQL-совместимость) на {self.host}:{self.port}")

            # Формируем параметры подключения
            conn_params = {
                'host': self.host,
                'port': self.port,
                'user': self.user,
                'password': self.password
            }
            # Добавляем database только если он задан
            if self.database:
                conn_params['dbname'] = self.database

            # Подключение
            self.connection = psycopg2.connect(**conn_params)

            # Проверка подключения
            with self.connection.cursor() as cur:
                cur.execute("SELECT version();")
                version_result = cur.fetchone()
                logger.success(f"Подключение к DuckDB (через psycopg2) успешно установлено")
                logger.debug(f"Версия сервера: {version_result[0] if version_result else 'Unknown'}")

            return True

        except psycopg2.Error as e: # Более конкретное исключение для psycopg2
            logger.error(f"Ошибка подключения к DuckDB (psycopg2): {e}")
            logger.error(f"Детали подключения: host={self.host}, port={self.port}, db={self.database}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при подключении к DuckDB: {e}")
            return False

    def connect(self, retries: int = 1, delay: int = 0) -> bool:
        """
        Установка соединения с базой данных с возможностью повторных попыток.

        Args:
            retries (int): Количество попыток подключения.
            delay (int): Задержка между попытками в секундах (не реализована в этом примере).

        Returns:
            bool: True если подключение успешно, False в противном случае.
        """
        # В простейшем случае, просто вызываем _attempt_connect
        # Можно добавить логику повтора здесь при необходимости
        return self._attempt_connect()

    def disconnect(self):
        """Закрытие соединения с базой данных"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None # Важно: сбросить ссылку
            logger.info("Соединение с базой данных закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения: {e}")

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Optional[List[Tuple[Any, ...]]]:
        """
        Выполнение SQL-запроса.

        Args:
            query (str): SQL-запрос.
            params (tuple, optional): Параметры для запроса.

        Returns:
            list: Результаты запроса (для SELECT/WITH) или True (для DML), None в случае ошибки.
        """
        try:
            if not self.connection:
                logger.error("Нет активного соединения с базой данных")
                return None

            logger.debug(f"Выполнение запроса: {query} с параметрами: {params}")

            with self.connection.cursor() as cursor:
                cursor.execute(query, params)

                # Проверяем тип запроса
                query_upper = query.strip().upper()
                if query_upper.startswith('SELECT') or 'WITH' in query_upper.split()[0:2]:
                    results = cursor.fetchall()
                    logger.debug(f"Запрос вернул {len(results) if results else 0} строк")
                    return results
                else:
                    # Для операций изменения данных фиксируем изменения
                    self.connection.commit()
                    logger.debug("Запрос выполнен, изменения зафиксированы")
                    return True # Или можно возвращать cursor.rowcount если нужно

        except psycopg2.Error as e: # Более конкретное исключение
            logger.error(f"Ошибка выполнения запроса (psycopg2): {e}")
            # Можно рассмотреть откат транзакции self.connection.rollback() если это критично
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при выполнении запроса: {e}")
            return None

    def preview_data(self, query: str, params: Optional[Tuple] = None, limit: int = 10):
        """
        Вывод превью данных с использованием tabulate.

        Args:
            query (str): SQL-запрос.
            params (tuple, optional): Параметры для запроса.
            limit (int): Количество строк для отображения (по умолчанию 10).
        """
        try:
             # Добавляем LIMIT к запросу если это SELECT
            if query.strip().upper().startswith('SELECT'):
                 # Используем подзапрос для корректного добавления LIMIT
                limited_query = f"SELECT * FROM ({query.rstrip(';')}) AS subquery LIMIT %s;"
                # Параметры запроса плюс limit
                final_params = (params if params else ()) + (limit,)
            else:
                limited_query = query
                final_params = params

            results = self.execute_query(limited_query, final_params)

            if results is not None and len(results) > 0:
                # Получаем названия колонок. Для этого нужно выполнить запрос с LIMIT 0
                # Но psycopg2.cursor.description доступен после execute.
                # Мы можем выполнить тот же ограниченный запрос и получить описание.
                # Или сделать отдельный запрос. Проще использовать результат execute_query.
                # Однако execute_query возвращает только данные, не описание.
                # Нужно немного изменить логику или сделать отдельный запрос для схемы.

                # Альтернатива: выполнить запрос с LIMIT 0 перед основным запросом
                # Но это не всегда работает корректно с подзапросами.
                # Лучше получить описание из курсора, использованного в execute_query.
                # Перепишем execute_query немного, чтобы она могла возвращать и описание.

                # Упрощённый вариант: получаем описание из отдельного запроса
                # Это не идеально, но работает для простых случаев.
                # Для более точного подхода execute_query нужно модифицировать.

                column_names = []
                # Выполняем тот же ограниченный запрос с LIMIT 0, чтобы получить структуру
                schema_query = f"SELECT * FROM ({query.rstrip(';')}) AS subquery LIMIT 0;"
                try:
                     with self.connection.cursor() as schema_cursor:
                        schema_cursor.execute(schema_query, params)
                        column_names = [desc[0] for desc in schema_cursor.description]
                except psycopg2.Error as e:
                    logger.warning(f"Не удалось получить названия колонок: {e}")
                    # Если не удалось, используем индексы или пустые заголовки
                    column_names = [f"Col_{i}" for i in range(len(results[0]))] if results else []

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
    logger.info("Запуск теста подключения к DuckDB (PostgreSQL-совместимость)")

    # Передаём параметры подключения явно или они будут взяты из config/по умолчанию
    with DuckDBConnector() as connector: # Использует параметры из __init__ и config
        logger.success("Тест подключения успешно пройден!")

        # Пример выполнения простого запроса
        logger.info("Выполнение тестового запроса...")
        connector.preview_data("SELECT version();")

        # Создадим тестовую таблицу и добавим данные
        # ВАЖНО: DuckDB в режиме PostgreSQL может требовать указания схемы (например, CREATE TABLE public.test_table ...)
        # или создание таблицы в текущей сессии (которая может быть временной, если не указана БД)
        logger.info("Создание тестовой таблицы...")
        # Проверим, существует ли таблица
        # connector.execute_query("DROP TABLE IF EXISTS test_table;") # Опционально, для чистоты теста
        create_result = connector.execute_query("""
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER,
            name VARCHAR
        );
        """)
        if create_result is True:
             logger.info("Таблица test_table создана или уже существует.")
        else:
             logger.error("Ошибка при создании таблицы test_table")

        insert_result = connector.execute_query("INSERT INTO test_table (id, name) VALUES (%s, %s), (%s, %s);", (1, 'Test', 2, 'Example'))
        if insert_result is True:
            logger.info("Данные вставлены в test_table.")
        else:
             logger.error("Ошибка при вставке данных в test_table")

        # Проверим данные
        connector.preview_data("SELECT * FROM test_table;")

        assert True


if __name__ == "__main__":
    test_connection()

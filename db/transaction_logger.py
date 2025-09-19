# transaction_logger.py
import sys
import os
from loguru import logger
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
import json

# Добавляем путь к коннектору
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    f"logs/transaction_logger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


class TransactionLogger:
    def __init__(self, strategy_name: str, trade_mode: str = "BACKTEST"):
        """
        Инициализация логгера транзакций

        Args:
            strategy_name (str): Название стратегии
            trade_mode (str): Режим торговли - "BACKTEST" или "LIVE"
        """
        self.strategy_name = strategy_name
        self.trade_mode = trade_mode.upper()
        if self.trade_mode not in ["BACKTEST", "LIVE"]:
            raise ValueError("trade_mode должен быть 'BACKTEST' или 'LIVE'")

        logger.info(f"📊 Инициализация TransactionLogger для стратегии '{strategy_name}' в режиме '{trade_mode}'")

    def log_transaction(self,
                        timestamp: datetime,
                        instrument_id: str,
                        trade_type: str,  # LONG или SHORT
                        action: str,  # OPEN или CLOSE
                        order_side: str,  # BUY или SELL
                        quantity: Decimal,
                        price: Decimal,
                        version: str, # Новое поле
                        order_id: Optional[str] = None,
                        position_id: Optional[str] = None,
                        realized_pnl: Optional[Decimal] = None,
                        commission: Decimal = Decimal('0'),
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Логирование транзакции в базу данных

        Args:
            timestamp: Время транзакции
            instrument_id: ID инструмента
            trade_type: Тип сделки (LONG/SHORT)
            action: Действие (OPEN/CLOSE)
            order_side: Сторона ордера (BUY/SELL)
            quantity: Количество
            price: Цена
            version: Версия (новый параметр)
            order_id: ID ордера
            position_id: ID позиции
            realized_pnl: Реализованный PnL
            commission: Комиссия
            metadata: Дополнительные данные в формате JSON

        Returns:
            bool: True если успешно, False если ошибка
        """
        # Добавлено поле version в список столбцов и в список параметров (%s)
        insert_query = """
        INSERT INTO transactions (
            timestamp, strategy_name, instrument_id, trade_type, action, 
            order_side, quantity, price, version, order_id, position_id, 
            realized_pnl, commission, trade_mode, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Добавлен version в кортеж параметров на соответствующую позицию
        params = (
            timestamp, self.strategy_name, instrument_id, trade_type.upper(), action.upper(),
            order_side.upper(), float(quantity), float(price), version, # Передаем version
            order_id, position_id,
            float(realized_pnl) if realized_pnl is not None else None,
            float(commission), self.trade_mode,
            json.dumps(metadata) if metadata else None
        )

        try:
            with PostgreSQLConnector("transactions") as db:
                if db.connection:
                    result = db.execute_query(insert_query, params)
                    if result is not None: # execute_query для INSERT возвращает True при успехе
                        logger.success(f"💾 Транзакция сохранена: {action} {trade_type} {instrument_id} @ {price} (v{version})")
                        return True
                    else:
                        logger.error(f"❌ Ошибка при сохранении транзакции: {instrument_id}")
                        return False
                else:
                    logger.error("❌ Нет подключения к базе данных")
                    return False
        except Exception as e:
            logger.error(f"❌ Исключение при сохранении транзакции: {e}")
            return False

    def log_open_position(self,
                          timestamp: datetime,
                          instrument_id: str,
                          trade_type: str,
                          order_side: str,
                          quantity: Decimal,
                          price: Decimal,
                          version: str, # Новое поле
                          order_id: Optional[str] = None,
                          position_id: Optional[str] = None,
                          commission: Decimal = Decimal('0'),
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Логирование открытия позиции"""
        logger.info(f"📈 Открытие позиции: {trade_type} {instrument_id} {quantity}@{price} (v{version})")
        # Передаем version в вызов log_transaction
        return self.log_transaction(
            timestamp=timestamp,
            instrument_id=instrument_id,
            trade_type=trade_type,
            action="OPEN",
            order_side=order_side,
            quantity=quantity,
            price=price,
            version=version, # Передаем version
            order_id=order_id,
            position_id=position_id,
            commission=commission,
            metadata=metadata
        )

    def log_close_position(self,
                           timestamp: datetime,
                           instrument_id: str,
                           trade_type: str,
                           order_side: str,
                           quantity: Decimal,
                           price: Decimal,
                           realized_pnl: Decimal,
                           version: str, # Новое поле
                           order_id: Optional[str] = None,
                           position_id: Optional[str] = None,
                           commission: Decimal = Decimal('0'),
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Логирование закрытия позиции"""
        pnl_status = "✅ ПРИБЫЛЬ" if realized_pnl >= 0 else "❌ УБЫТОК"
        logger.info(
            f"📉 Закрытие позиции: {trade_type} {instrument_id} {quantity}@{price} | {pnl_status}: {realized_pnl} (v{version})")

        # Передаем version в вызов log_transaction
        return self.log_transaction(
            timestamp=timestamp,
            instrument_id=instrument_id,
            trade_type=trade_type,
            action="CLOSE",
            order_side=order_side,
            quantity=quantity,
            price=price,
            version=version, # Передаем version
            order_id=order_id,
            position_id=position_id,
            realized_pnl=realized_pnl,
            commission=commission,
            metadata=metadata
        )

    def get_transactions_summary(self, limit: int = 10) -> None:
        """Получение сводки последних транзакций"""
        # Добавлено version в SELECT
        query = """
        SELECT 
            timestamp,
            instrument_id,
            trade_type,
            action,
            order_side,
            quantity,
            price,
            version, -- Добавлено поле version
            realized_pnl,
            trade_mode
        FROM transactions 
        WHERE strategy_name = %s 
        ORDER BY timestamp DESC 
        LIMIT %s
        """

        try:
            with PostgreSQLConnector("transactions") as db:
                if db.connection:
                    db.preview_data(query, (self.strategy_name, limit))
                else:
                    logger.error("❌ Нет подключения к базе данных")
        except Exception as e:
            logger.error(f"❌ Ошибка при получении сводки: {e}")

    def get_pnl_summary(self) -> None:
        """Получение сводки по прибыли/убыткам"""
        # Этот запрос не меняется, так как version не используется в агрегации
        query = """
        SELECT 
            COUNT(*) as total_trades,
            COUNT(CASE WHEN realized_pnl >= 0 THEN 1 END) as winning_trades,
            COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
            SUM(realized_pnl) as total_pnl,
            AVG(realized_pnl) as avg_pnl,
            MAX(realized_pnl) as max_win,
            MIN(realized_pnl) as max_loss
        FROM transactions 
        WHERE strategy_name = %s AND action = 'CLOSE' AND realized_pnl IS NOT NULL
        """

        try:
            with PostgreSQLConnector("transactions") as db:
                if db.connection:
                    results = db.execute_query(query, (self.strategy_name,))
                    if results:
                        row = results[0]
                        logger.info("💰 Сводка по прибыли/убыткам:")
                        logger.info(f"   Всего закрытых сделок: {row[0]}")
                        logger.info(f"   Прибыльных: {row[1]} | Убыточных: {row[2]}")
                        logger.info(f"   Общая прибыль: {row[3]:.2f}")
                        logger.info(f"   Средняя прибыль: {row[4]:.2f}")
                        logger.info(f"   Макс. выигрыш: {row[5]:.2f} | Макс. потеря: {row[6]:.2f}")
                else:
                    logger.error("❌ Нет подключения к базе данных")
        except Exception as e:
            logger.error(f"❌ Ошибка при получении сводки PnL: {e}")

# Пример использования (если скрипт запускается напрямую)
if __name__ == "__main__":
    # Пример использования TransactionLogger
    logger.info("Пример использования TransactionLogger")

    # Создаем экземпляр логгера
    transaction_logger = TransactionLogger(strategy_name="MyStrategy_v1.0", trade_mode="BACKTEST")

    # Логируем открытие позиции
    transaction_logger.log_open_position(
        timestamp=datetime.now(),
        instrument_id="BTC/USD",
        trade_type="LONG",
        order_side="BUY",
        quantity=Decimal('1.5'),
        price=Decimal('45000.0'),
        version="1.0.0" # Передаем версию
    )

    # Логируем закрытие позиции
    transaction_logger.log_close_position(
        timestamp=datetime.now(),
        instrument_id="BTC/USD",
        trade_type="LONG",
        order_side="SELL",
        quantity=Decimal('1.5'),
        price=Decimal('46000.0'),
        realized_pnl=Decimal('1500.0'),
        version="1.0.0" # Передаем версию
    )

    # Получаем сводку
    transaction_logger.get_transactions_summary()

    # Получаем сводку PnL
    transaction_logger.get_pnl_summary()

"""
Tinkoff execution client.
"""

import asyncio
from typing import Dict, List, Optional

from nautilus_trader.adapters.tinkoff.client import TinkoffClient
from nautilus_trader.adapters.tinkoff.instruments import TinkoffInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.common.uuid import UUID4
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.execution.client import ExecutionClient
from nautilus_trader.execution.reports import OrderStatusReport
from nautilus_trader.execution.reports import PositionStatusReport
from nautilus_trader.execution.reports import TradeReport
from nautilus_trader.model.commands import CancelAllOrders
from nautilus_trader.model.commands import CancelOrder
from nautilus_trader.model.commands import ModifyOrder
from nautilus_trader.model.commands import SubmitOrder
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import ContingencyType
from nautilus_trader.model.enums import LiquiditySide
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderState
from nautilus_trader.model.enums import OrderType
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import TrailingOffsetType
from nautilus_trader.model.enums import TriggerType
from nautilus_trader.model.identifiers import AccountId
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import VenueOrderId
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.msgbus.bus import MessageBus

from nautilus_trader.adapters.tinkoff.common import TINKOFF_VENUE
from nautilus_trader.adapters.tinkoff.config import TinkoffExecClientConfig
from nautilus_trader.adapters.tinkoff.schemas import TinkoffOrder
from nautilus_trader.adapters.tinkoff.schemas import TinkoffPosition
from nautilus_trader.adapters.tinkoff.schemas import TinkoffTrade


class TinkoffLiveExecutionClient(ExecutionClient):
    """
    Tinkoff client for live execution.
    """

    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            client: TinkoffClient,
            msgbus: MessageBus,
            cache: Cache,
            clock: LiveClock,
            logger: Logger,
            instrument_provider: TinkoffInstrumentProvider,
            exec_config: TinkoffExecClientConfig,
    ):
        super().__init__(
            loop=loop,
            client_id=ClientId(f"TINKOFF-{id(self)}"),
            venue=TINKOFF_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=None,  # Tinkoff поддерживает несколько валют
            instrument_provider=instrument_provider,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            logger=logger,
        )

        self._client = client
        self._config = exec_config
        self._instrument_provider = instrument_provider

        # Словари для отслеживания ордеров
        self._order_client_to_venue = {}  # ClientOrderId -> VenueOrderId
        self._order_venue_to_client = {}  # VenueOrderId -> ClientOrderId

        # Устанавливаем обработчики событий
        self._client._event_handlers["on_portfolio"] = self._on_portfolio
        self._client._event_handlers["on_position"] = self._on_position
        self._client._event_handlers["on_order_trades"] = self._on_order_trades

    async def _connect(self):
        """Connect to the Tinkoff API."""
        self._log.info("Connecting to Tinkoff execution API...")
        await self._client.connect()
        self._log.info("Connected to Tinkoff execution API")

        # Загружаем инструменты
        await self._instrument_provider.load_all()

        # Запрашиваем текущие ордера
        await self._request_active_orders()

    async def _disconnect(self):
        """Disconnect from the Tinkoff API."""
        self._log.info("Disconnecting from Tinkoff execution API...")
        await self._client.disconnect()
        self._log.info("Disconnected from Tinkoff execution API")

    async def _request_active_orders(self):
        """Запрашиваем активные ордера."""
        self._log.debug("Requesting active orders...")

        # Получаем активные ордера через TinkoffPy
        orders = self._client._provider.call_function(
            self._client._provider.stub_orders.GetOrders,
            request={"account_id": self._config.account_id}
        )

        # Преобразуем и публикуем статусы ордеров
        for order in orders:
            nautilus_order = self._tinkoff_order_to_nautilus(order)
            if nautilus_order:
                self._send_order_status_report(nautilus_order)

    async def submit_order(self, command: SubmitOrder):
        """Submit an order to the exchange."""
        order = command.order
        self._log.debug(f"Submitting order: {order}")

        # Получаем информацию об инструменте
        instrument = self._cache.instrument(order.instrument_id)
        if not instrument:
            self._log.error(f"Instrument {order.instrument_id} not found")
            return

        # Преобразуем ордер Nautilus в формат Tinkoff
        tinkoff_order = self._nautilus_order_to_tinkoff(order, instrument)

        try:
            # Отправляем ордер
            response = self._client._provider.call_function(
                self._client._provider.stub_orders.PostOrder,
                request=tinkoff_order
            )

            # Сохраняем соответствие ClientOrderId -> VenueOrderId
            self._order_client_to_venue[order.client_order_id] = VenueOrderId(response.order_id)
            self._order_venue_to_client[response.order_id] = order.client_order_id

            # Публикуем подтверждение ордера
            self.generate_order_submitted(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                ts_event=self._clock.timestamp_ns(),
            )

            self._log.info(f"Order submitted: {order.client_order_id} -> {response.order_id}")

        except Exception as e:
            self._log.error(f"Error submitting order: {e}")
            self.generate_order_rejected(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                reason=str(e),
                ts_event=self._clock.timestamp_ns(),
            )

    async def modify_order(self, command: ModifyOrder):
        """Modify an existing order."""
        order = self._cache.order(command.client_order_id)
        if not order:
            self._log.error(f"Order {command.client_order_id} not found")
            return

        self._log.debug(f"Modifying order: {order}")

        # Получаем информацию об инструменте
        instrument = self._cache.instrument(order.instrument_id)
        if not instrument:
            self._log.error(f"Instrument {order.instrument_id} not found")
            return

        # Преобразуем ордер Nautilus в формат Tinkoff
        tinkoff_order = self._nautilus_order_to_tinkoff(order, instrument, command)

        try:
            # Модифицируем ордер
            response = self._client._provider.call_function(
                self._client._provider.stub_orders.ReplaceOrder,
                request=tinkoff_order
            )

            # Обновляем соответствие ClientOrderId -> VenueOrderId
            # При модификации может быть создан новый ордер
            new_venue_id = VenueOrderId(response.order_id)
            self._order_client_to_venue[command.client_order_id] = new_venue_id
            self._order_venue_to_client[new_venue_id] = command.client_order_id

            # Публикуем подтверждение модификации
            self.generate_order_updated(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=command.client_order_id,
                venue_order_id=new_venue_id,
                quantity=command.quantity,
                price=command.price,
                trigger_price=command.trigger_price,
                ts_event=self._clock.timestamp_ns(),
            )

            self._log.info(f"Order modified: {command.client_order_id} -> {response.order_id}")

        except Exception as e:
            self._log.error(f"Error modifying order: {e}")
            self.generate_order_rejected(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=command.client_order_id,
                reason=str(e),
                ts_event=self._clock.timestamp_ns(),
            )

    async def cancel_order(self, command: CancelOrder):
        """Cancel an existing order."""
        order = self._cache.order(command.client_order_id)
        if not order:
            self._log.error(f"Order {command.client_order_id} not found")
            return

        self._log.debug(f"Cancelling order: {order}")

        try:
            # Отменяем ордер
            response = self._client._provider.call_function(
                self._client._provider.stub_orders.CancelOrder,
                request={
                    "account_id": self._config.account_id,
                    "order_id": self._order_client_to_venue[command.client_order_id].value
                }
            )

            # Публикуем подтверждение отмены
            self.generate_order_canceled(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=command.client_order_id,
                venue_order_id=self._order_client_to_venue[command.client_order_id],
                ts_event=self._clock.timestamp_ns(),
            )

            self._log.info(f"Order cancelled: {command.client_order_id}")

        except Exception as e:
            self._log.error(f"Error cancelling order: {e}")
            self.generate_order_rejected(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=command.client_order_id,
                reason=str(e),
                ts_event=self._clock.timestamp_ns(),
            )

    async def cancel_all_orders(self, command: CancelAllOrders):
        """Cancel all orders for a strategy."""
        # Получаем все ордера для стратегии
        orders = self._cache.orders_for_venue(
            venue=TINKOFF_VENUE,
            strategy_id=command.strategy_id,
        )

        for order in orders:
            await self.cancel_order(
                CancelOrder(
                    client_order_id=order.client_order_id,
                    venue_order_id=self._order_client_to_venue.get(order.client_order_id),
                    command_id=UUID4(),
                    command_timestamp=self._clock.timestamp_ns(),
                )
            )

    async def generate_order_status_report(
            self,
            instrument_id: InstrumentId,
            client_order_id: Optional[ClientOrderId] = None,
            venue_order_id: Optional[VenueOrderId] = None,
    ) -> Optional[OrderStatusReport]:
        """Generate an order status report."""
        self._log.debug(f"Generating order status report for {client_order_id or venue_order_id}")

        # Получаем информацию об инструменте
        instrument = self._cache.instrument(instrument_id)
        if not instrument:
            self._log.error(f"Instrument {instrument_id} not found")
            return None

        try:
            # Запрашиваем статус ордера
            order = self._client._provider.call_function(
                self._client._provider.stub_orders.GetOrderState,
                request={
                    "account_id": self._config.account_id,
                    "order_id": venue_order_id.value if venue_order_id else self._order_client_to_venue[
                        client_order_id].value
                }
            )

            # Преобразуем в отчет
            return self._tinkoff_order_to_report(order, instrument)

        except Exception as e:
            self._log.error(f"Error generating order status report: {e}")
            return None

    async def generate_order_status_reports(
            self,
            instrument_id: Optional[InstrumentId] = None,
            start: Optional[dt] = None,
            end: Optional[dt] = None,
    ) -> List[OrderStatusReport]:
        """Generate order status reports."""
        self._log.debug("Generating order status reports")

        reports = []

        try:
            # Получаем все активные ордера
            orders = self._client._provider.call_function(
                self._client._provider.stub_orders.GetOrders,
                request={"account_id": self._config.account_id}
            )

            # Преобразуем каждый ордер в отчет
            for order in orders:
                instrument = self._cache.instrument(
                    InstrumentId(Symbol(order.symbol), TINKOFF_VENUE)
                )
                if not instrument:
                    continue

                report = self._tinkoff_order_to_report(order, instrument)
                if report:
                    reports.append(report)

            return reports

        except Exception as e:
            self._log.error(f"Error generating order status reports: {e}")
            return []

    async def generate_position_status_reports(
            self,
            instrument_id: Optional[InstrumentId] = None,
            start: Optional[dt] = None,
            end: Optional[dt] = None,
    ) -> List[PositionStatusReport]:
        """Generate position status reports."""
        self._log.debug("Generating position status reports")

        reports = []

        try:
            # Получаем позиции
            positions = self._client._provider.call_function(
                self._client._provider.stub_operations.GetPositions,
                request={"account_id": self._config.account_id}
            )

            # Преобразуем каждую позицию в отчет
            for position in positions:
                instrument = self._cache.instrument(
                    InstrumentId(Symbol(position.symbol), TINKOFF_VENUE)
                )
                if not instrument:
                    continue

                report = self._tinkoff_position_to_report(position, instrument)
                if report:
                    reports.append(report)

            return reports

        except Exception as e:
            self._log.error(f"Error generating position status reports: {e}")
            return []

    def _on_portfolio(self, portfolio):
        """Обработка обновления портфеля."""
        self._log.debug(f"Portfolio update received: {portfolio}")

        # Публикуем баланс счета
        for position in portfolio.positions:
            instrument = self._cache.instrument(
                InstrumentId(Symbol(position.symbol), TINKOFF_VENUE)
            )
            if not instrument:
                continue

            self.generate_account_state(
                account_id=AccountId(f"TINKOFF-{self._config.account_id}"),
                balances=[],
                margins=[],
                reported=True,
                ts_event=self._clock.timestamp_ns(),
            )

    def _on_position(self, position):
        """Обработка обновления позиции."""
        self._log.debug(f"Position update received: {position}")

        # Публикуем обновление позиции
        instrument = self._cache.instrument(
            InstrumentId(Symbol(position.symbol), TINKOFF_VENUE)
        )
        if not instrument:
            return

        # Определяем сторону позиции
        side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL

        # Создаем отчет о позиции
        report = PositionStatusReport(
            account_id=AccountId(f"TINKOFF-{self._config.account_id}"),
            instrument_id=instrument.id,
            position_side=side,
            quantity=Quantity(abs(position.quantity), instrument.quantity_precision),
            ts_last=self._clock.timestamp_ns(),
            report_id=UUID4(),
            ts_init=self._clock.timestamp_ns(),
        )

        self._send_position_status_report(report)

    def _on_order_trades(self, order_trades):
        """Обработка сделок по заявкам."""
        self._log.debug(f"Order trades received: {order_trades}")

        # Публикуем исполнение сделок
        for trade in order_trades.trades:
            # Находим ордер
            client_order_id = self._order_venue_to_client.get(trade.order_id)
            if not client_order_id:
                continue

            order = self._cache.order(client_order_id)
            if not order:
                continue

            # Получаем инструмент
            instrument = self._cache.instrument(order.instrument_id)
            if not instrument:
                continue

            # Преобразуем цену
            price = self._client._provider.tinkoff_price_to_price(
                instrument.raw_symbol.value,
                instrument.symbol.value,
                trade.price
            )

            # Создаем отчет о сделке
            report = TradeReport(
                account_id=AccountId(f"TINKOFF-{self._config.account_id}"),
                instrument_id=instrument.id,
                execution_id=TradeId(trade.id),
                client_order_id=client_order_id,
                venue_order_id=VenueOrderId(trade.order_id),
                quantity=Quantity(trade.quantity, instrument.quantity_precision),
                price=Price(price, instrument.price_precision),
                commission=Money(0, instrument.quote_currency),
                liquidity_side=LiquiditySide.TAKER,
                report_id=UUID4(),
                ts_event=dt_to_unix_nanos(trade.time),
                ts_init=self._clock.timestamp_ns(),
            )

            self._handle_trade_report(report)

    def _tinkoff_order_to_nautilus(self, tinkoff_order) -> Optional[Order]:
        """Преобразует ордер Tinkoff в формат Nautilus."""
        # Получаем инструмент
        instrument_id = InstrumentId(
            Symbol(tinkoff_order.symbol),
            TINKOFF_VENUE
        )
        instrument = self._cache.instrument(instrument_id)
        if not instrument:
            return None

        # Определяем тип ордера
        order_type = OrderType.LIMIT
        if tinkoff_order.type == "MARKET":
            order_type = OrderType.MARKET
        elif tinkoff_order.type == "STOP":
            order_type = OrderType.STOP_MARKET

        # Создаем ClientOrderId
        client_order_id = ClientOrderId(f"TINKOFF-{tinkoff_order.order_id}")

        # Сохраняем соответствие
        self._order_client_to_venue[client_order_id] = VenueOrderId(tinkoff_order.order_id)
        self._order_venue_to_client[tinkoff_order.order_id] = client_order_id

        # Создаем ордер
        order = self._get_order_factory().create_order(
            order_type=order_type,
            instrument_id=instrument_id,
            client_order_id=client_order_id,
            quantity=Quantity(tinkoff_order.quantity, instrument.quantity_precision),
            time_in_force=TimeInForce.GTC,
            init_id=UUID4(),
            ts_init=self._clock.timestamp_ns(),
        )

        # Устанавливаем дополнительные параметры
        if order_type == OrderType.LIMIT:
            price = self._client._provider.tinkoff_price_to_price(
                instrument.raw_symbol.value,
                instrument.symbol.value,
                tinkoff_order.price
            )
            order.price = Price(price, instrument.price_precision)

        if order_type == OrderType.STOP_MARKET:
            stop_price = self._client._provider.tinkoff_price_to_price(
                instrument.raw_symbol.value,
                instrument.symbol.value,
                tinkoff_order.stop_price
            )
            order.trigger_price = Price(stop_price, instrument.price_precision)

        # Устанавливаем сторону
        order.side = OrderSide.BUY if tinkoff_order.direction == "BUY" else OrderSide.SELL

        # Устанавливаем статус
        if tinkoff_order.status == "NEW":
            order.apply(
                OrderState.SUBMITTED,
                self._clock.timestamp_ns(),
                self._clock.timestamp_ns(),
            )
        elif tinkoff_order.status == "PARTIALLY_FILLED":
            order.apply(
                OrderState.PARTIALLY_FILLED,
                self._clock.timestamp_ns(),
                self._clock.timestamp_ns(),
            )
            order.filled = Quantity(tinkoff_order.executed_quantity, instrument.quantity_precision)
        elif tinkoff_order.status == "FILLED":
            order.apply(
                OrderState.FILLED,
                self._clock.timestamp_ns(),
                self._clock.timestamp_ns(),
            )
            order.filled = Quantity(tinkoff_order.quantity, instrument.quantity_precision)
        elif tinkoff_order.status in ["CANCELLED", "REJECTED"]:
            order.apply(
                OrderState.CANCELED,
                self._clock.timestamp_ns(),
                self._clock.timestamp_ns(),
            )

        return order

    def _nautilus_order_to_tinkoff(self, order, instrument, modify_command=None) -> Dict:
        """Преобразует ордер Nautilus в формат Tinkoff."""
        # Определяем тип ордера
        order_type = "LIMIT"
        if order.order_type == OrderType.MARKET:
            order_type = "MARKET"
        elif order.order_type == OrderType.STOP_MARKET:
            order_type = "STOP"

        # Преобразуем цену
        price = None
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_MARKET]:
            price = self._client._provider.price_to_tinkoff_price(
                instrument.raw_symbol.value,
                instrument.symbol.value,
                order.price.as_double()
            )

        # Преобразуем стоп-цену
        stop_price = None
        if order.order_type == OrderType.STOP_MARKET:
            stop_price = self._client._provider.price_to_tinkoff_price(
                instrument.raw_symbol.value,
                instrument.symbol.value,
                order.trigger_price.as_double()
            )

        # Определяем направление
        direction = "BUY" if order.side == OrderSide.BUY else "SELL"

        # Создаем запрос
        request = {
            "account_id": self._config.account_id,
            "instrument_id": instrument.raw_symbol.value,
            "quantity": int(order.quantity),
            "direction": direction,
            "order_type": order_type,
        }

        # Добавляем цену, если нужно
        if price is not None:
            request["price"] = price

        # Добавляем стоп-цену, если нужно
        if stop_price is not None:
            request["stop_price"] = stop_price

        # Если это модификация ордера
        if modify_command:
            request["order_id"] = self._order_client_to_venue[order.client_order_id].value

            # Обновляем цену, если указана
            if modify_command.price:
                request["price"] = self._client._provider.price_to_tinkoff_price(
                    instrument.raw_symbol.value,
                    instrument.symbol.value,
                    modify_command.price.as_double()
                )

            # Обновляем количество, если указано
            if modify_command.quantity:
                request["quantity"] = int(modify_command.quantity)

        return request

    def _tinkoff_order_to_report(self, tinkoff_order, instrument) -> OrderStatusReport:
        """Преобразует ордер Tinkoff в отчет о статусе."""
        # Создаем ClientOrderId
        client_order_id = ClientOrderId(f"TINKOFF-{tinkoff_order.order_id}")

        # Определяем статус
        status = OrderState.SUBMITTED
        if tinkoff_order.status == "PARTIALLY_FILLED":
            status = OrderState.PARTIALLY_FILLED
        elif tinkoff_order.status == "FILLED":
            status = OrderState.FILLED
        elif tinkoff_order.status in ["CANCELLED", "REJECTED"]:
            status = OrderState.CANCELED

        # Создаем отчет
        return OrderStatusReport(
            account_id=AccountId(f"TINKOFF-{self._config.account_id}"),
            instrument_id=instrument.id,
            client_order_id=client_order_id,
            venue_order_id=VenueOrderId(tinkoff_order.order_id),
            order_side=OrderSide.BUY if tinkoff_order.direction == "BUY" else OrderSide.SELL,
            order_type=OrderType.LIMIT,  # TODO: определить тип ордера
            quantity=Quantity(tinkoff_order.quantity, instrument.quantity_precision),
            filled_qty=Quantity(tinkoff_order.executed_quantity, instrument.quantity_precision),
            avg_px=None,  # TODO: заполнить средней ценой
            status=status,
            report_id=UUID4(),
            ts_accepted=dt_to_unix_nanos(tinkoff_order.created_at),
            ts_last=dt_to_unix_nanos(tinkoff_order.updated_at),
            ts_init=self._clock.timestamp_ns(),
        )

    def _tinkoff_position_to_report(self, tinkoff_position, instrument) -> PositionStatusReport:
        """Преобразует позицию Tinkoff в отчет о статусе."""
        # Определяем сторону позиции
        side = OrderSide.BUY if tinkoff_position.quantity > 0 else OrderSide.SELL

        # Создаем отчет
        return PositionStatusReport(
            account_id=AccountId(f"TINKOFF-{self._config.account_id}"),
            instrument_id=instrument.id,
            position_side=side,
            quantity=Quantity(abs(tinkoff_position.quantity), instrument.quantity_precision),
            ts_last=self._clock.timestamp_ns(),
            report_id=UUID4(),
            ts_init=self._clock.timestamp_ns(),
        )
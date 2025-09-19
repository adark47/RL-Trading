# t_adapter/execution_client.py

# -*- coding: utf-8 -*-
"""
Tinkoff Invest adapter execution client.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import uuid

from tinkoff.invest import Client, RequestError
from tinkoff.invest.schemas import (
    OrderDirection,
    OrderType,
    PostOrderRequest,
    PostOrderResponse,
    CancelOrderRequest,
    GetOrdersRequest,
    GetOrdersResponse,
    GetOrderStateRequest,
    OrderState,
    OrderExecutionReportStatus,
    PortfolioRequest,
    PortfolioResponse,
    PositionsRequest,
    PositionsResponse,
    Quotation,
)

from nautilus_trader.execution.messages import (
    SubmitOrder,
    ModifyOrder,
    CancelOrder,
    BatchCancelOrders,
    CancelAllOrders,
)
from nautilus_trader.execution.reports import OrderStatusReport, FillReport, PositionStatusReport
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.enums import (
    OrderSide,
    OrderType as NautilusOrderType,
    OrderStatus,
    TimeInForce,
    ContingencyType,
    TriggerType,
    OrderSide,
)
from nautilus_trader.model.identifiers import (
    AccountId,
    ClientOrderId,
    InstrumentId,
    TradeId,
    VenueOrderId,
)
from nautilus_trader.model.objects import Quantity, Price, Money
from nautilus_trader.model.orders import Order
from nautilus_trader.msgbus import MessageBus
from nautilus_trader.cache import Cache
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.datetime import millis_to_nanos
from nautilus_trader.core.uuid import UUID4

from t_adapter.common import TINKOFF_VENUE
from t_adapter.instrument_provider import TAdapterInstrumentProvider
from t_adapter.config import TAdapterExecClientConfig


class TAdapterExecutionClient(LiveExecutionClient):
    """
    Provides an execution client for Tinkoff Invest.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client: Client,
        account_id: AccountId, # Specific account this client manages
        msgbus: MessageBus,
        cache: Cache,
        clock: ...,
        instrument_provider: TAdapterInstrumentProvider,
        config: TAdapterExecClientConfig,
        name: Optional[str] = None,
    ):
        """
        Initialize a new instance of the ``TAdapterExecutionClient`` class.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        client : Client
            The Tinkoff Invest client instance.
        account_id : AccountId
            The account ID associated with this execution client.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.
        instrument_provider : TAdapterInstrumentProvider
            The instrument provider.
        config : TAdapterExecClientConfig
            The configuration for the client.
        name : str, optional
            The custom client ID.
        """
        super().__init__(
            loop=loop,
            client_id=account_id, # Use AccountId as client ID
            venue=TINKOFF_VENUE,
            oms_type=..., # Define based on Tinkoff's OMS (usually NETTING or FIFO)
            account_type=..., # Define based on account type (CASH, MARGIN, etc.)
            base_currency=..., # Define base currency or get from account
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
            name=name,
        )

        self._client = client
        self._account_id = account_id
        self._is_sandbox = config.is_sandbox
        # Extract Tinkoff account ID string if needed (from AccountId.value)
        self._tinkoff_account_id = config.account_id or self._get_default_account_id()

        # Internal state
        self._venue_order_id_to_client_order_id: Dict[VenueOrderId, ClientOrderId] = {}
        self._client_order_id_to_venue_order_id: Dict[ClientOrderId, VenueOrderId] = {}

    def _get_default_account_id(self) -> str:
        """
        Get the default account ID from Tinkoff Invest API if not configured.

        Returns
        -------
        str
            The default account ID.
        """
        try:
            accounts_response = self._client.users.get_accounts()
            if accounts_response.accounts:
                # Return the first account ID as default
                # A more sophisticated selection logic might be needed
                return accounts_response.accounts[0].id
            else:
                self._log.error("No accounts found for Tinkoff Invest user")
                return ""
        except RequestError as e:
            self._log.error(f"Error getting default account ID: {e}")
            return ""

    async def _connect(self) -> None:
        """
        Connects the client to Tinkoff Invest.
        """
        self._log.info(f"Connecting to Tinkoff Invest execution for account {self._account_id}", LogColor.BLUE)
        try:
            # Connection is managed by the invest-python Client.
            # Verify account access
            account_id_str = self._tinkoff_account_id
            if not account_id_str:
                raise RuntimeError("Tinkoff account ID is not configured or could not be retrieved.")

            # A simple check to see if account is accessible
            self._client.users.get_info() # Basic call to check connectivity
            self._log.info(f"Connected to Tinkoff Invest execution for account {self._account_id}", LogColor.GREEN)
        except Exception as e:
            self._log.error(f"Error connecting to Tinkoff Invest execution: {e}")
            raise

    async def _disconnect(self) -> None:
        """
        Disconnects the client from Tinkoff Invest.
        """
        self._log.info(f"Disconnecting from Tinkoff Invest execution for account {self._account_id}", LogColor.BLUE)
        try:
            # No explicit disconnect needed for the Client context if managed externally
            # If internal async client is used, it should be closed here
            self._log.info(f"Disconnected from Tinkoff Invest execution for account {self._account_id}", LogColor.GREEN)
        except Exception as e:
            self._log.error(f"Error disconnecting from Tinkoff Invest execution: {e}")

    # Order Execution Methods
    async def _submit_order(self, command: SubmitOrder) -> None:
        """
        Submit an order.

        Parameters
        ----------
        command : SubmitOrder
            The command to submit an order.
        """
        PyCondition.not_none(command, "command")

        self._log.info(f"Submitting order {command}", LogColor.BLUE)

        try:
            instrument = self._cache.instrument(command.instrument_id)
            if not instrument:
                self._log.error(f"Cannot submit order: instrument {command.instrument_id} not found in cache")
                return

            # Extract order details
            client_order_id = command.order.client_order_id
            instrument_id = command.instrument_id
            order_side = command.order.side
            order_type = command.order.order_type
            quantity = command.order.quantity
            price = command.order.price
            time_in_force = command.order.time_in_force
            # Reduce-only, post-only etc. need to be checked against Tinkoff capabilities

            # Map Nautilus types to Tinkoff types
            figi = instrument_id.symbol.value # Assuming symbol is FIGI

            tinkoff_direction = OrderDirection.ORDER_DIRECTION_BUY if order_side == OrderSide.BUY else OrderDirection.ORDER_DIRECTION_SELL

            tinkoff_order_type = OrderType.ORDER_TYPE_MARKET
            if order_type == NautilusOrderType.LIMIT:
                tinkoff_order_type = OrderType.ORDER_TYPE_LIMIT
            elif order_type == NautilusOrderType.STOP_MARKET:
                tinkoff_order_type = OrderType.ORDER_TYPE_STOP # Check if this is correct mapping
            elif order_type == NautilusOrderType.STOP_LIMIT:
                tinkoff_order_type = OrderType.ORDER_TYPE_STOP_LIMIT # Check mapping
            # Add other order types as supported by Tinkoff

            # Convert Quantity to integer lots for Tinkoff
            lots = int(quantity.as_double() / instrument.lot_size) # Assuming quantity is in units, convert to lots
            if lots <= 0:
                 self._log.error(f"Invalid order quantity {quantity} for {instrument_id}. Calculated lots: {lots}")
                 return

            # Convert Price to Quotation for Tinkoff
            price_quotation = None
            if price is not None:
                # Price is a Decimal. Convert to units and nano.
                price_value = price.as_decimal()
                units = int(price_value)
                nano = int((price_value - units) * 1_000_000_000)
                price_quotation = Quotation(units=units, nano=nano)

            # TimeInForce mapping (Tinkoff might have limited TIF options)
            # Tinkoff common TIFs: DAY, GTC, IOC, FOK
            # Nautilus: GTC, IOC, FOK, GTD, DAY
            tinkoff_tif = None # Tinkoff API might not explicitly use TIF in PostOrderRequest in the same way
            # It might be implicit based on order type or expiry settings.
            # Check Tinkoff docs for correct way to specify TIF.

            # Build the request
            request = PostOrderRequest(
                figi=figi,
                quantity=lots,
                price=price_quotation,
                direction=tinkoff_direction,
                account_id=self._tinkoff_account_id,
                order_type=tinkoff_order_type,
                order_id=client_order_id.value, # Use Nautilus ClientOrderId as Tinkoff order_id
                # time_in_force=tinkoff_tif, # Add if applicable
                # instrument_id=... # Might be needed depending on Tinkoff API version
            )

            # Submit the order
            response: PostOrderResponse = self._client.orders.post_order(request)

            # Process the response
            venue_order_id = VenueOrderId(response.order_id)
            # Map venue order ID back to client order ID for later reference
            self._venue_order_id_to_client_order_id[venue_order_id] = client_order_id
            self._client_order_id_to_venue_order_id[client_order_id] = venue_order_id

            # Determine initial status from response
            status = OrderStatus.SUBMITTED
            if response.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL:
                status = OrderStatus.FILLED
            elif response.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED:
                status = OrderStatus.REJECTED
            # Add other status mappings

            # Create and send order accepted report
            self.generate_order_accepted(
                strategy_id=command.strategy_id,
                instrument_id=instrument_id,
                client_order_id=client_order_id,
                venue_order_id=venue_order_id,
                ts_event=millis_to_nanos(response.order_request_id) if response.order_request_id else self._clock.timestamp_ns(), # Use appropriate timestamp
            )

            # If filled immediately, send fill report
            if status == OrderStatus.FILLED:
                 # Convert executed quantity and price
                 executed_lots = response.lots_executed
                 executed_qty = Quantity(executed_lots * instrument.lot_size, precision=instrument.size_precision)
                 executed_price = self._quotation_to_price(response.executed_order_price, instrument.price_precision) if response.executed_order_price else None

                 # Create and send fill report
                 self.generate_order_filled(
                     strategy_id=command.strategy_id,
                     instrument_id=instrument_id,
                     client_order_id=client_order_id,
                     venue_order_id=venue_order_id,
                     venue_position_id=None, # Tinkoff might not provide this directly
                     trade_id=TradeId(str(uuid.uuid4())), # Generate or use Tinkoff trade ID if available
                     order_side=order_side,
                     order_type=order_type,
                     last_qty=executed_qty,
                     last_px=executed_price,
                     quote_currency=instrument.quote_currency,
                     commission=Money(0, instrument.quote_currency), # Get actual commission if available
                     liquidity_side=..., # Determine aggressor side if possible
                     ts_event=millis_to_nanos(response.order_request_id) if response.order_request_id else self._clock.timestamp_ns(),
                 )

            self._log.info(f"Submitted order {client_order_id} with venue ID {venue_order_id}", LogColor.GREEN)

        except RequestError as e:
            # Handle Tinkoff API errors
            self._log.error(f"Error submitting order {command}: {e}")
            # Send order rejected report
            self.generate_order_rejected(
                strategy_id=command.strategy_id,
                instrument_id=command.instrument_id,
                client_order_id=command.order.client_order_id,
                reason=str(e),
                ts_event=self._clock.timestamp_ns(),
            )
        except Exception as e:
             self._log.exception(f"Unexpected error submitting order {command}: {e}")
             # Send order rejected report for unexpected errors
             self.generate_order_rejected(
                 strategy_id=command.strategy_id,
                 instrument_id=command.instrument_id,
                 client_order_id=command.order.client_order_id,
                 reason=f"Unexpected error: {e}",
                 ts_event=self._clock.timestamp_ns(),
             )

    async def _modify_order(self, command: ModifyOrder) -> None:
        """
        Modify an existing order. Tinkoff API might not support direct order modification.
        Common practice is to cancel and replace.
        """
        self._log.warning(f"Order modification not directly supported by Tinkoff API. Please cancel and resubmit order {command.client_order_id}.", LogColor.WARNING)
        # If modification is crucial, implement cancel + submit logic here
        # based on the `command` details.

    async def _cancel_order(self, command: CancelOrder) -> None:
        """
        Cancel an existing order.

        Parameters
        ----------
        command : CancelOrder
            The command to cancel an order.
        """
        PyCondition.not_none(command, "command")

        self._log.info(f"Cancelling order {command}", LogColor.BLUE)

        try:
            client_order_id = command.client_order_id
            venue_order_id = command.venue_order_id

            # Need venue_order_id to cancel. If not provided, try to find it.
            if not venue_order_id:
                venue_order_id = self._client_order_id_to_venue_order_id.get(client_order_id)

            if not venue_order_id:
                self._log.error(f"Cannot cancel order {client_order_id}: Venue order ID not found")
                return

            # Build cancel request
            request = CancelOrderRequest(
                account_id=self._tinkoff_account_id,
                order_id=venue_order_id.value,
            )

            # Cancel the order
            self._client.orders.cancel_order(request)

            # Tinkoff cancel_order doesn't return a status response in the same way.
            # We assume success if no exception is raised.
            # A subsequent order status check might be needed to confirm cancellation.

            self._log.info(f"Cancelled order {client_order_id} (Venue ID: {venue_order_id})", LogColor.GREEN)

        except RequestError as e:
            self._log.error(f"Error cancelling order {command.client_order_id}: {e}")
            # Depending on error, might want to query order status to confirm state
        except Exception as e:
             self._log.exception(f"Unexpected error cancelling order {command.client_order_id}: {e}")

    async def _batch_cancel_orders(self, command: BatchCancelOrders) -> None:
        """
        Batch cancel orders.
        Tinkoff API might require cancelling orders one by one or has a specific batch method.
        We'll iterate and cancel individually.
        """
        self._log.info(f"Batch cancelling orders {command.client_order_ids}", LogColor.BLUE)

        for client_order_id in command.client_order_ids:
            # Create a CancelOrder command for each
            cancel_cmd = CancelOrder(
                trader_id=command.trader_id,
                strategy_id=command.strategy_id,
                instrument_id=command.instrument_id, # Might be None, handle accordingly
                client_order_id=client_order_id,
                venue_order_id=None, # Let _cancel_order find it
                command_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
            )
            await self._cancel_order(cancel_cmd)

        self._log.info(f"Batch cancellation initiated for {len(command.client_order_ids)} orders", LogColor.GREEN)

    async def _cancel_all_orders(self, command: CancelAllOrders) -> None:
        """
        Cancel all orders for a specific instrument or all instruments.

        Parameters
        ----------
        command : CancelAllOrders
            The command to cancel all orders.
        """
        self._log.info(f"Cancelling all orders for instrument {command.instrument_id}", LogColor.BLUE)

        try:
            # Tinkoff provides an endpoint to cancel all orders for an account
            # This might be limited to the current session or all open orders.
            # Check Tinkoff API docs for exact behavior.
            # For now, we'll use the get_orders to find open orders and cancel them.

            # Get all open orders for the account (and optionally filter by instrument)
            request = GetOrdersRequest(
                account_id=self._tinkoff_account_id,
                # instrument_id=... # Add if filtering by specific instrument FIGI
            )
            response: GetOrdersResponse = self._client.orders.get_orders(request)

            cancelled_count = 0
            for order in response.orders:
                # Check if order is open/cancellable
                # OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW
                # OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL
                if order.execution_report_status in [OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW,
                                                     OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL]:
                    try:
                        cancel_request = CancelOrderRequest(
                            account_id=self._tinkoff_account_id,
                            order_id=order.order_id,
                        )
                        self._client.orders.cancel_order(cancel_request)
                        cancelled_count += 1
                        # Update internal maps if necessary
                        venue_order_id = VenueOrderId(order.order_id)
                        client_order_id = self._venue_order_id_to_client_order_id.get(venue_order_id)
                        if client_order_id:
                             self._client_order_id_to_venue_order_id.pop(client_order_id, None)
                             self._venue_order_id_to_client_order_id.pop(venue_order_id, None)

                    except RequestError as e:
                        self._log.warning(f"Error cancelling order {order.order_id}: {e}")
                    except Exception as e:
                         self._log.exception(f"Unexpected error cancelling order {order.order_id}: {e}")

            self._log.info(f"Cancelled {cancelled_count} orders for account {self._account_id}", LogColor.GREEN)

        except RequestError as e:
            self._log.error(f"Error getting orders list for cancellation: {e}")
        except Exception as e:
             self._log.exception(f"Unexpected error cancelling all orders: {e}")

    # Report Generation Methods
    async def generate_order_status_report(self, command: ... ) -> OrderStatusReport:
        """
        Generate an order status report.
        This is usually called in response to a specific request or periodically.
        """
        self._log.info(f"Generating order status report for {command.client_order_id}", LogColor.BLUE)

        try:
            client_order_id = command.client_order_id
            venue_order_id = command.venue_order_id

            if not venue_order_id:
                venue_order_id = self._client_order_id_to_venue_order_id.get(client_order_id)

            if not venue_order_id:
                self._log.error(f"Cannot generate order status report: Venue order ID not found for {client_order_id}")
                return None

            # Get order state from Tinkoff
            request = GetOrderStateRequest(
                account_id=self._tinkoff_account_id,
                order_id=venue_order_id.value,
            )
            response: OrderState = self._client.orders.get_order_state(request)

            # Map Tinkoff OrderState to Nautilus OrderStatusReport
            instrument_id = InstrumentId.from_str(f"{response.figi}.{TINKOFF_VENUE.value}") # Need to map FIGI correctly
            instrument = self._cache.instrument(instrument_id)
            if not instrument:
                 self._log.warning(f"Instrument {instrument_id} not found in cache for order status report")
                 # Create a minimal instrument or proceed with caution

            order_side = OrderSide.BUY if response.direction == OrderDirection.ORDER_DIRECTION_BUY else OrderSide.SELL
            order_type = NautilusOrderType.MARKET # Default, refine based on response.type
            if response.order_type == OrderType.ORDER_TYPE_LIMIT:
                 order_type = NautilusOrderType.LIMIT
            elif response.order_type == OrderType.ORDER_TYPE_STOP:
                 order_type = NautilusOrderType.STOP_MARKET
            # Add other type mappings

            status = OrderStatus.ACCEPTED # Default
            if response.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL:
                status = OrderStatus.FILLED
            elif response.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED:
                status = OrderStatus.REJECTED
            elif response.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_CANCELLED:
                status = OrderStatus.CANCELED
            elif response.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW:
                 status = OrderStatus.ACCEPTED
            elif response.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL:
                 status = OrderStatus.PARTIALLY_FILLED

            price = self._quotation_to_price(response.average_position_price, instrument.price_precision) if response.average_position_price else None
            if not price and response.initial_order_price:
                 price = self._quotation_to_price(response.initial_order_price, instrument.price_precision)

            quantity = Quantity(response.lots_requested * instrument.lot_size, precision=instrument.size_precision) if instrument else Quantity(response.lots_requested, precision=0)
            filled_qty = Quantity(response.lots_executed * instrument.lot_size, precision=instrument.size_precision) if instrument else Quantity(response.lots_executed, precision=0)

            report = OrderStatusReport(
                account_id=self._account_id,
                instrument_id=instrument_id,
                client_order_id=client_order_id,
                order_list_id=None,
                venue_order_id=venue_order_id,
                order_side=order_side,
                order_type=order_type,
                contingency_type=ContingencyType.NO_CONTINGENCY, # Check if Tinkoff supports OCO/OTO
                trigger_type=TriggerType.NO_TRIGGER, # Check if applicable
                time_in_force=TimeInForce.GTC, # Refine based on order details
                expire_time=None, # Add if order has expiry
                order_status=status,
                price=price,
                trigger_price=None, # Add if stop order
                quantity=quantity,
                filled_qty=filled_qty,
                avg_px=self._quotation_to_price(response.average_position_price, instrument.price_precision) if response.average_position_price and instrument else None,
                post_only=False, # Determine from order details if available
                reduce_only=False, # Determine from order details if available
                cancel_reason=response.cancel_reason if response.cancel_reason else None,
                ts_accepted=millis_to_nanos(response.created_at.seconds * 1000 + response.created_at.nanos // 1_000_000) if response.created_at else 0,
                ts_last=millis_to_nanos(response.updated_at.seconds * 1000 + response.updated_at.nanos // 1_000_000) if response.updated_at else 0,
                ts_init=self._clock.timestamp_ns(),
            )

            self._log.info(f"Generated order status report for {client_order_id}", LogColor.GREEN)
            return report

        except RequestError as e:
            self._log.error(f"Error generating order status report for {command.client_order_id}: {e}")
            return None
        except Exception as e:
             self._log.exception(f"Unexpected error generating order status report for {command.client_order_id}: {e}")
             return None

    async def generate_fill_reports(self, command: ...) -> List[FillReport]:
        """
        Generate fill reports. This might involve querying trade history.
        Tinkoff API has methods to get operations (trades) within a period.
        """
        # This is a placeholder. Implement based on Tinkoff's trade history API.
        # Usually involves GetOperations or similar endpoint.
        self._log.info("Generating fill reports (not implemented in detail)", LogColor.BLUE)
        return [] # Return list of FillReport objects

    async def generate_position_status_reports(self, command: ...) -> List[PositionStatusReport]:
        """
        Generate position status reports.
        This involves querying the portfolio and positions.
        """
        self._log.info("Generating position status reports", LogColor.BLUE)

        reports = []
        try:
            # Get portfolio
            portfolio_request = PortfolioRequest(account_id=self._tinkoff_account_id)
            portfolio_response: PortfolioResponse = self._client.operations.get_portfolio(portfolio_request)

            # Get positions
            positions_request = PositionsRequest(account_id=self._tinkoff_account_id)
            positions_response: PositionsResponse = self._client.operations.get_positions(positions_request)

            # Process positions response
            for position in positions_response.securities:
                # Map position data to PositionStatusReport
                # position.figi, position.balance, position.blocked, etc.
                instrument_id = InstrumentId.from_str(f"{position.figi}.{TINKOFF_VENUE.value}")
                instrument = self._cache.instrument(instrument_id)
                if not instrument:
                     self._log.warning(f"Instrument {instrument_id} not found in cache for position report")
                     continue # Skip or create minimal instrument

                net_qty = Quantity(position.balance, precision=instrument.size_precision) # Assuming balance is in units
                locked_qty = Quantity(position.blocked, precision=instrument.size_precision)

                report = PositionStatusReport(
                    account_id=self._account_id,
                    instrument_id=instrument_id,
                    net_qty=net_qty,
                    locked_qty=locked_qty,
                    # avg_px_open, side, etc. might require more detailed portfolio analysis or are not directly available
                    ts_last=self._clock.timestamp_ns(), # Use appropriate timestamp
                    ts_init=self._clock.timestamp_ns(),
                )
                reports.append(report)

            # Process currency positions if needed
            for currency_position in positions_response.money:
                 # Create a report or handle differently if needed
                 pass

            self._log.info(f"Generated {len(reports)} position status reports", LogColor.GREEN)

        except RequestError as e:
            self._log.error(f"Error generating position status reports: {e}")
        except Exception as e:
             self._log.exception(f"Unexpected error generating position status reports: {e}")

        return reports

    def _quotation_to_price(self, quotation: Quotation, price_precision: int) -> Price:
        """
        Convert a Tinkoff Quotation to a Nautilus Price.

        Parameters
        ----------
        quotation : Quotation
            The Tinkoff Quotation proto.
        price_precision : int
            The price precision for the instrument.

        Returns
        -------
        Price
        """
        from decimal import Decimal
        value = Decimal(quotation.units) + Decimal(quotation.nano) / Decimal(1_000_000_000)
        return Price(value, precision=price_precision)

# t_adapter/data_client.py

# -*- coding: utf-8 -*-
"""
Tinkoff Invest adapter data client.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import datetime

from tinkoff.invest import Client, RequestError, AsyncClient
from tinkoff.invest.schemas import (
    SubscriptionAction,
    SubscriptionInterval,
    CandleInterval,
    TradeDirection,
    OrderBook,
    Trade,
    Candle,
    MarketDataResponse,
    GetCandlesRequest,
    GetCandlesResponse,
    Quotation,
)

from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import Bar, BarType, QuoteTick, TradeTick, OrderBookSnapshot, OrderBookDelta, OrderBookDeltas
from nautilus_trader.model.enums import BookType, AggressorSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.msgbus import MessageBus
from nautilus_trader.cache import Cache
from nautilus_trader.core.datetime import millis_to_nanos, secs_to_nanos
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.live.data_client import LiveMarketDataClient

from t_adapter.common import TINKOFF_VENUE
from t_adapter.instrument_provider import TAdapterInstrumentProvider
from t_adapter.enums import TinkoffProductType


class TAdapterDataClient(LiveMarketDataClient):
    """
    Provides market data feeds from Tinkoff Invest.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client: Client, # Assuming a synchronous client is passed, or we manage async internally
        msgbus: MessageBus,
        cache: Cache,
        clock: ...,
        instrument_provider: TAdapterInstrumentProvider,
        config: ...,
        name: Optional[str] = None,
    ):
        """
        Initialize a new instance of the ``TAdapterDataClient`` class.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        client : Client
            The Tinkoff Invest client instance (sync or managed async).
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.
        instrument_provider : TAdapterInstrumentProvider
            The instrument provider.
        config : TAdapterDataClientConfig
            The configuration for the client.
        name : str, optional
            The custom client ID.
        """
        super().__init__(
            loop=loop,
            client_id=Venue(name or TINKOFF_VENUE.value), # Use venue as client ID
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
        )

        self._client = client
        self._is_sandbox = config.is_sandbox
        # Assuming the client is already configured with the token and target (sandbox/production)
        # If not, it should be configured before passing to this client.

        # Internal state for subscriptions
        self._book_subscriptions: Dict[InstrumentId, bool] = {}
        self._trade_subscriptions: Dict[InstrumentId, bool] = {}
        self._ticker_subscriptions: Dict[InstrumentId, bool] = {} # Assuming ticker means last trade price, handled via trades
        self._candle_subscriptions: Dict[BarType, bool] = {}

        # For handling async stream if needed (though invest-python handles it)
        # We might need an async task to listen to the stream
        self._stream_task: Optional[asyncio.Task] = None
        self._stream_active = False

    async def _connect(self) -> None:
        """
        Connects the client to Tinkoff Invest.
        """
        self._log.info("Connecting to Tinkoff Invest market data stream")
        try:
            # The connection is managed by the invest-python Client context.
            # If we need to explicitly start a stream listener, we do it here.
            # The Client should already be initialized with the token.
            self._stream_active = True
            self._stream_task = asyncio.create_task(self._listen_to_stream())
            self._log.info("Connected to Tinkoff Invest market data stream")
        except Exception as e:
            self._log.error(f"Error connecting to Tinkoff Invest market data stream: {e}")
            self._stream_active = False

    async def _disconnect(self) -> None:
        """
        Disconnects the client from Tinkoff Invest.
        """
        self._log.info("Disconnecting from Tinkoff Invest market data stream")
        try:
            self._stream_active = False
            if self._stream_task:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            # Close the Tinkoff client if it has a close method or context needs exiting
            # If it's passed as a context-managed object, closing might be handled externally
            self._log.info("Disconnected from Tinkoff Invest market data stream")
        except Exception as e:
            self._log.error(f"Error disconnecting from Tinkoff Invest market data stream: {e}")

    async def _listen_to_stream(self):
        """
        Listen to the Tinkoff Invest market data stream and process messages.
        This is a key part of the data client.
        """
        try:
            # Using the async client context for streaming
            # The sync client might not support streaming directly in the same way
            # We'll assume an async client is available or can be created
            # For simplicity, let's assume `self._client` can be used for streaming
            # If it's sync, we'd need to manage the async client separately or use threading

            # This example assumes an async approach or that the sync client has streaming capabilities
            # The actual implementation might require creating an AsyncClient within this method
            # or managing it separately.

            # Placeholder for actual stream listening logic
            # This is a simplified representation. The actual stream handling requires
            # calling the appropriate Tinkoff Invest streaming methods and processing the yielded responses.

            # Example using AsyncClient (conceptual, needs correct import and setup)
            # async with AsyncClient(token) as async_client:
            #     async for marketdata in async_client.market_data_stream.market_data_stream(...):
            #         await self._process_market_data_response(marketdata)

            # Since the prompt specifies not to implement http_client/websocket_client,
            # and invest-python handles the connection, we assume `self._client` (or an internal
            # async version of it) can be used to subscribe and listen.

            # This is a conceptual loop. The actual implementation depends on how
            # invest-python exposes the streaming API.
            while self._stream_active:
                # This is a placeholder. In reality, you'd iterate over the stream responses
                # from Tinkoff Invest's market data API.
                # The invest-python library usually handles this iteration internally
                # when you call methods like `market_data_stream.subscribe_candles`.
                # Therefore, the logic here would involve calling the subscription methods
                # and then having callbacks or a way to process the yielded data.

                # A more realistic approach would be to subscribe to instruments as needed
                # in the subscribe_* methods and let the library handle the streaming loop.
                # This _listen_to_stream method might not be strictly necessary if the library
                # manages the loop, but it's common to have one to control the lifecycle.

                # For now, we'll simulate a wait. In practice, this loop would be driven
                # by the async stream.
                await asyncio.sleep(0.1) # Prevent busy waiting

        except asyncio.CancelledError:
            self._log.debug("Market data stream listener task cancelled.")
        except RequestError as e:
            self._log.error(f"Error in Tinkoff Invest market data stream: {e}")
        except Exception as e:
            self._log.exception(f"Unexpected error in market data stream listener: {e}")
        finally:
            self._log.info("Market data stream listener stopped.")

    def _process_market_data_response(self, response: MarketDataResponse):
        """
        Process a single MarketDataResponse message from the stream.

        Parameters
        ----------
        response : MarketDataResponse
            The market data response message.
        """
        try:
            # Dispatch based on the type of data received
            if response.HasField('subscribe_candles_response'):
                self._log.debug(f"Received candle subscription response: {response.subscribe_candles_response}")
            elif response.HasField('subscribe_order_book_response'):
                self._log.debug(f"Received order book subscription response: {response.subscribe_order_book_response}")
            elif response.HasField('subscribe_trades_response'):
                self._log.debug(f"Received trades subscription response: {response.subscribe_trades_response}")
            elif response.HasField('candle'):
                self._handle_candle(response.candle)
            elif response.HasField('trade'):
                self._handle_trade(response.trade)
            elif response.HasField('orderbook'):
                self._handle_order_book(response.orderbook)
            elif response.HasField('ping'):
                self._log.debug("Received ping from Tinkoff Invest stream")
                # Might need to send a pong, but invest-python likely handles this
            elif response.HasField('error'):
                self._log.error(f"Received error from Tinkoff Invest stream: {response.error}")
            # Add handling for other message types like ` trading_status`, `ping`, etc.
            else:
                self._log.warning(f"Received unhandled market data response type: {response}")

        except Exception as e:
            self._log.exception(f"Error processing market data response: {e}")

    def _handle_candle(self, candle: Candle):
        """
        Handle a received candle and send it to the Nautilus message bus.

        Parameters
        ----------
        candle : Candle
            The Tinkoff Invest Candle proto.
        """
        try:
            # Map Tinkoff candle interval to Nautilus BarType
            # This requires knowing which BarType subscription this candle belongs to
            # We need to track subscriptions and map intervals
            # For simplicity, let's assume we can derive BarType from the candle info
            # This is a simplification; in practice, you'd need a mapping from
            # subscribed figi+interval to BarType.

            figi = candle.fig_i # Note: Check the actual field name in proto
            interval = candle.interval

            # Find the corresponding BarType subscription (this is tricky without a direct map)
            # A better approach is to store the mapping when subscribing
            # Let's assume we have a way to get the BarType from the subscription context
            # For now, we'll need to infer or have a pre-existing mapping
            # This is a limitation of this simplified example.

            # Get instrument from cache or provider
            instrument = self._cache.instrument(InstrumentId.from_str(f"{figi}.{TINKOFF_VENUE.value}"))
            if not instrument:
                self._log.warning(f"Received candle for unknown instrument {figi}")
                return

            bar_type = None # Need to find the correct BarType based on subscription
            # This is a placeholder. In a real implementation, you'd have a mapping
            # from (figi, interval) to subscribed BarType.
            # Example: self._candle_subscription_map.get((figi, interval))
            # For now, we'll skip processing without a clear BarType mapping.

            if bar_type is None:
                self._log.warning(f"Could not determine BarType for candle {candle}")
                return

            # Convert Tinkoff Quotation to Nautilus Price
            open_price = self._quotation_to_price(candle.open, instrument.price_precision)
            high_price = self._quotation_to_price(candle.high, instrument.price_precision)
            low_price = self._quotation_to_price(candle.low, instrument.price_precision)
            close_price = self._quotation_to_price(candle.close, instrument.price_precision)

            # Volume is typically in lots for Tinkoff. Convert to instrument units.
            # `volume` in candle is int64. It represents the number of lots traded.
            # `instrument.lot_size` tells us how many units are in a lot.
            volume_qty = Quantity(candle.volume * instrument.lot_size, precision=instrument.size_precision)

            # Timestamps: Tinkoff uses google.protobuf.timestamp_pb2.Timestamp
            # Nautilus uses nanoseconds since epoch.
            ts_event = int(candle.time.seconds * 1_000_000_000 + candle.time.nanos)
            ts_init = self._clock.timestamp_ns() # When we received/processed it

            bar = Bar(
                bar_type=bar_type,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume_qty,
                ts_event=ts_event,
                ts_init=ts_init,
            )

            # Send the bar to the message bus
            self._handle_data(bar)

        except Exception as e:
            self._log.exception(f"Error handling candle {candle}: {e}")

    def _handle_trade(self, trade: Trade):
        """
        Handle a received trade and send it as a TradeTick.

        Parameters
        ----------
        trade : Trade
            The Tinkoff Invest Trade proto.
        """
        try:
            figi = trade.fig_i # Check field name
            instrument_id = InstrumentId.from_str(f"{figi}.{TINKOFF_VENUE.value}")
            instrument = self._cache.instrument(instrument_id)
            if not instrument:
                self._log.warning(f"Received trade for unknown instrument {figi}")
                return

            # Convert price and size
            price = self._quotation_to_price(trade.price, instrument.price_precision)
            # Trade quantity in Tinkoff is usually in lots. Convert to units.
            size = Quantity(trade.quantity * instrument.lot_size, precision=instrument.size_precision)

            # Aggressor side
            aggressor_side = AggressorSide.BUYER if trade.direction == TradeDirection.TRADE_DIRECTION_BUY else AggressorSide.SELLER

            # Trade ID: Tinkoff might provide a trade ID. If not, we might need to generate one
            # or use timestamp + sequence if available. Using timestamp for now.
            # trade_id = TradeId(str(trade.trade_id)) if trade.HasField('trade_id') and trade.trade_id else TradeId(str(trade.time.seconds * 1000000000 + trade.time.nanos))

            # Timestamps
            ts_event = int(trade.time.seconds * 1_000_000_000 + trade.time.nanos)
            ts_init = self._clock.timestamp_ns()

            trade_tick = TradeTick(
                instrument_id=instrument_id,
                price=price,
                size=size,
                aggressor_side=aggressor_side,
                # trade_id=trade_id, # Uncomment if trade_id is reliably available
                ts_event=ts_event,
                ts_init=ts_init,
            )

            self._handle_data(trade_tick)

        except Exception as e:
            self._log.exception(f"Error handling trade {trade}: {e}")

    def _handle_order_book(self, order_book: OrderBook):
        """
        Handle a received order book snapshot.

        Parameters
        ----------
        order_book : OrderBook
            The Tinkoff Invest OrderBook proto.
        """
        try:
            figi = order_book.fig_i # Check field name
            instrument_id = InstrumentId.from_str(f"{figi}.{TINKOFF_VENUE.value}")
            instrument = self._cache.instrument(instrument_id)
            if not instrument:
                self._log.warning(f"Received order book for unknown instrument {figi}")
                return

            # Tinkoff provides bids and asks as lists of Order proto (price, quantity)
            # Nautilus OrderBookSnapshot expects lists of (price, size) tuples

            bids = [
                (
                    self._quotation_to_price(bid.price, instrument.price_precision),
                    Quantity(bid.quantity * instrument.lot_size, precision=instrument.size_precision) # Convert lots to units
                )
                for bid in order_book.bids
            ]
            asks = [
                (
                    self._quotation_to_price(ask.price, instrument.price_precision),
                    Quantity(ask.quantity * instrument.lot_size, precision=instrument.size_precision) # Convert lots to units
                )
                for ask in order_book.asks
            ]

            # Timestamps
            ts_event = int(order_book.time.seconds * 1_000_000_000 + order_book.time.nanos)
            ts_init = self._clock.timestamp_ns()

            book_snapshot = OrderBookSnapshot(
                instrument_id=instrument_id,
                book_type=BookType.L2_MBP, # Assuming Tinkoff provides L2 data
                bids=bids,
                asks=asks,
                ts_event=ts_event,
                ts_init=ts_init,
            )

            self._handle_data(book_snapshot)

        except Exception as e:
            self._log.exception(f"Error handling order book {order_book}: {e}")

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
        # Quotation has units (int64) and nano (int32) parts.
        # The actual value is units + nano / 1,000,000,000
        # Nautilus Price expects a decimal value.
        # We can use Decimal or rely on Nautilus's internal conversion if it accepts float.
        # Using Decimal for precision.
        from decimal import Decimal
        value = Decimal(quotation.units) + Decimal(quotation.nano) / Decimal(1_000_000_000)
        return Price(value, precision=price_precision)

    async def _subscribe_order_book_snapshots(
        self,
        instrument_id: InstrumentId,
        book_type: BookType,
        depth: int = 5,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Subscribe to order book snapshot data for a given instrument.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to subscribe to.
        book_type : BookType
            The type of book to subscribe to.
        depth : int, default 5
            The book depth to subscribe to.
        params : dict[str, Any], optional
            Additional parameters for the subscription.
        """
        # Extract FIGI from instrument_id (assuming format FIGI.VENUE)
        figi = instrument_id.symbol.value # This might need adjustment based on actual symbol format

        if figi in self._book_subscriptions:
            self._log.warning(f"Already subscribed to order book snapshots for {instrument_id}")
            return

        self._log.info(f"Subscribing to order book snapshots for {instrument_id} with depth {depth}")

        try:
             # Use the Tinkoff client to subscribe to order book
             # This requires calling the appropriate method on the market_data_stream
             # Example (conceptual, check invest-python docs for exact method):
             # await self._client.market_data_stream.subscribe_order_book(figi, depth, SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE)

             # Placeholder for actual subscription call
             # self._client.market_data_stream.subscribe_order_book(...) # This needs to be async if client supports it, or managed via the stream context

             # For now, mark as subscribed
             self._book_subscriptions[instrument_id] = True
             self._log.info(f"Subscribed to order book snapshots for {instrument_id}")

        except RequestError as e:
            self._log.error(f"Error subscribing to order book snapshots for {instrument_id}: {e}")
        except Exception as e:
            self._log.exception(f"Unexpected error subscribing to order book snapshots for {instrument_id}: {e}")

    async def _subscribe_order_book_deltas(
        self,
        instrument_id: InstrumentId,
        book_type: BookType,
        depth: int = 5,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Tinkoff Invest API primarily provides snapshots. Subscribing to deltas might not be directly supported
        in the same way as other venues. We might need to diff snapshots or rely on the snapshot stream.
        For now, we'll treat this similar to snapshots or raise a not supported error.
        """
        self._log.warning("_subscribe_order_book_deltas not directly supported by Tinkoff Invest API. Subscribing to snapshots instead.")
        await self._subscribe_order_book_snapshots(instrument_id, book_type, depth, params)

    async def _subscribe_trade_ticks(
        self,
        instrument_id: InstrumentId,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Subscribe to trade tick data for a given instrument.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to subscribe to.
        params : dict[str, Any], optional
            Additional parameters for the subscription.
        """
        figi = instrument_id.symbol.value # Adjust based on symbol format

        if figi in self._trade_subscriptions:
            self._log.warning(f"Already subscribed to trade ticks for {instrument_id}")
            return

        self._log.info(f"Subscribing to trade ticks for {instrument_id}")

        try:
            # Use the Tinkoff client to subscribe to trades
            # Example (conceptual):
            # await self._client.market_data_stream.subscribe_trades(figi, SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE)

            # Placeholder for actual subscription call
            # self._client.market_data_stream.subscribe_trades(...)

            # Mark as subscribed
            self._trade_subscriptions[instrument_id] = True
            self._log.info(f"Subscribed to trade ticks for {instrument_id}")

        except RequestError as e:
            self._log.error(f"Error subscribing to trade ticks for {instrument_id}: {e}")
        except Exception as e:
             self._log.exception(f"Unexpected error subscribing to trade ticks for {instrument_id}: {e}")

    async def _subscribe_bars(
        self,
        bar_type: BarType,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Subscribe to bar data for a given bar type.

        Parameters
        ----------
        bar_type : BarType
            The bar type to subscribe to.
        params : dict[str, Any], optional
            Additional parameters for the subscription.
        """
        instrument_id = bar_type.instrument_id
        figi = instrument_id.symbol.value # Adjust based on symbol format

        # Map Nautilus BarType aggregation to Tinkoff CandleInterval
        # This mapping needs to be precise
        interval_map = {
            1: CandleInterval.CANDLE_INTERVAL_1_MIN,
            2: CandleInterval.CANDLE_INTERVAL_2_MIN,
            3: CandleInterval.CANDLE_INTERVAL_3_MIN,
            5: CandleInterval.CANDLE_INTERVAL_5_MIN,
            10: CandleInterval.CANDLE_INTERVAL_10_MIN,
            15: CandleInterval.CANDLE_INTERVAL_15_MIN,
            30: CandleInterval.CANDLE_INTERVAL_30_MIN,
            60: CandleInterval.CANDLE_INTERVAL_HOUR,
            120: CandleInterval.CANDLE_INTERVAL_2_HOUR,
            240: CandleInterval.CANDLE_INTERVAL_4_HOUR,
            1440: CandleInterval.CANDLE_INTERVAL_DAY,
            10080: CandleInterval.CANDLE_INTERVAL_WEEK,
            2592000: CandleInterval.CANDLE_INTERVAL_MONTH, # Approximation
        }

        aggregation = bar_type.spec.aggregation
        interval = interval_map.get(aggregation)
        if not interval:
            self._log.error(f"Unsupported bar aggregation {aggregation} for {bar_type}")
            return

        if bar_type in self._candle_subscriptions:
            self._log.warning(f"Already subscribed to bars for {bar_type}")
            return

        self._log.info(f"Subscribing to bars for {bar_type}")

        try:
            # Use the Tinkoff client to subscribe to candles
            # Example (conceptual):
            # await self._client.market_data_stream.subscribe_candles(figi, interval, SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE)

            # Placeholder for actual subscription call
            # self._client.market_data_stream.subscribe_candles(...)

            # Store subscription mapping
            # self._candle_subscription_map[(figi, interval)] = bar_type # If using mapping

            # Mark as subscribed
            self._candle_subscriptions[bar_type] = True
            self._log.info(f"Subscribed to bars for {bar_type}")

        except RequestError as e:
            self._log.error(f"Error subscribing to bars for {bar_type}: {e}")
        except Exception as e:
             self._log.exception(f"Unexpected error subscribing to bars for {bar_type}: {e}")

    # Implement unsubscribe methods similarly
    async def _unsubscribe_order_book_snapshots(self, instrument_id: InstrumentId) -> None:
        """Unsubscribe from order book snapshot data."""
        figi = instrument_id.symbol.value
        if figi not in self._book_subscriptions:
             self._log.warning(f"Not subscribed to order book snapshots for {instrument_id}")
             return

        self._log.info(f"Unsubscribing from order book snapshots for {instrument_id}")
        try:
            # Call unsubscribe method
            # await self._client.market_data_stream.unsubscribe_order_book(figi, SubscriptionAction.SUBSCRIPTION_ACTION_UNSUBSCRIBE)
            # Placeholder
            self._book_subscriptions.pop(instrument_id, None)
            self._log.info(f"Unsubscribed from order book snapshots for {instrument_id}")
        except RequestError as e:
            self._log.error(f"Error unsubscribing from order book snapshots for {instrument_id}: {e}")
        except Exception as e:
             self._log.exception(f"Unexpected error unsubscribing from order book snapshots for {instrument_id}: {e}")

    async def _unsubscribe_order_book_deltas(self, instrument_id: InstrumentId) -> None:
        """Unsubscribe from order book delta data."""
        # Treat same as snapshots for now
        await self._unsubscribe_order_book_snapshots(instrument_id)

    async def _unsubscribe_trade_ticks(self, instrument_id: InstrumentId) -> None:
        """Unsubscribe from trade tick data."""
        figi = instrument_id.symbol.value
        if figi not in self._trade_subscriptions:
             self._log.warning(f"Not subscribed to trade ticks for {instrument_id}")
             return

        self._log.info(f"Unsubscribing from trade ticks for {instrument_id}")
        try:
            # Call unsubscribe method
            # await self._client.market_data_stream.unsubscribe_trades(figi, SubscriptionAction.SUBSCRIPTION_ACTION_UNSUBSCRIBE)
            # Placeholder
            self._trade_subscriptions.pop(instrument_id, None)
            self._log.info(f"Unsubscribed from trade ticks for {instrument_id}")
        except RequestError as e:
            self._log.error(f"Error unsubscribing from trade ticks for {instrument_id}: {e}")
        except Exception as e:
             self._log.exception(f"Unexpected error unsubscribing from trade ticks for {instrument_id}: {e}")

    async def _unsubscribe_bars(self, bar_type: BarType) -> None:
        """Unsubscribe from bar data."""
        instrument_id = bar_type.instrument_id
        figi = instrument_id.symbol.value

        # Find interval from bar_type (reverse of subscription mapping)
        # interval = ... # Derive from bar_type.spec.aggregation

        if bar_type not in self._candle_subscriptions:
             self._log.warning(f"Not subscribed to bars for {bar_type}")
             return

        self._log.info(f"Unsubscribing from bars for {bar_type}")
        try:
            # Call unsubscribe method
            # await self._client.market_data_stream.unsubscribe_candles(figi, interval, SubscriptionAction.SUBSCRIPTION_ACTION_UNSUBSCRIBE)
            # Placeholder
            self._candle_subscriptions.pop(bar_type, None)
            # Also remove from mapping if used
            # self._candle_subscription_map.pop((figi, interval), None)
            self._log.info(f"Unsubscribed from bars for {bar_type}")
        except RequestError as e:
            self._log.error(f"Error unsubscribing from bars for {bar_type}: {e}")
        except Exception as e:
             self._log.exception(f"Unexpected error unsubscribing from bars for {bar_type}: {e}")

    # Implement historical data fetching if needed (e.g., for backtesting setup)
    async def _request_bars(
        self,
        bar_type: BarType,
        start: datetime,
        end: datetime,
        limit: int,
        correlation_id: UUID4,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Request historical bar data.
        """
        self._log.info(f"Requesting historical bars for {bar_type} from {start} to {end}")

        try:
            instrument_id = bar_type.instrument_id
            figi = instrument_id.symbol.value

            # Map BarType to Tinkoff interval
            interval_map = {
                1: CandleInterval.CANDLE_INTERVAL_1_MIN,
                2: CandleInterval.CANDLE_INTERVAL_2_MIN,
                3: CandleInterval.CANDLE_INTERVAL_3_MIN,
                5: CandleInterval.CANDLE_INTERVAL_5_MIN,
                10: CandleInterval.CANDLE_INTERVAL_10_MIN,
                15: CandleInterval.CANDLE_INTERVAL_15_MIN,
                30: CandleInterval.CANDLE_INTERVAL_30_MIN,
                60: CandleInterval.CANDLE_INTERVAL_HOUR,
                120: CandleInterval.CANDLE_INTERVAL_2_HOUR,
                240: CandleInterval.CANDLE_INTERVAL_4_HOUR,
                1440: CandleInterval.CANDLE_INTERVAL_DAY,
                10080: CandleInterval.CANDLE_INTERVAL_WEEK,
                2592000: CandleInterval.CANDLE_INTERVAL_MONTH,
            }

            aggregation = bar_type.spec.aggregation
            interval = interval_map.get(aggregation)
            if not interval:
                self._log.error(f"Unsupported bar aggregation {aggregation} for historical data request {bar_type}")
                return

            # Convert datetime to Tinkoff timestamp format if needed, or use string format
            # Tinkoff API might accept datetime objects or specific string formats
            # from_dt = start.isoformat() + "Z" # Example format
            # to_dt = end.isoformat() + "Z"

            # Prepare request
            request = GetCandlesRequest(
                figi=figi,
                from_=start, # Check field name, might be 'from'
                to=end,
                interval=interval,
                limit=limit,
            )

            # Make the request using the client
            response: GetCandlesResponse = self._client.market_data.get_candles(request)

            # Process the response and send bars
            instrument = self._cache.instrument(instrument_id)
            if not instrument:
                self._log.warning(f"Received historical bars for unknown instrument {instrument_id}")
                return

            bars = []
            for candle in response.candles:
                open_price = self._quotation_to_price(candle.open, instrument.price_precision)
                high_price = self._quotation_to_price(candle.high, instrument.price_precision)
                low_price = self._quotation_to_price(candle.low, instrument.price_precision)
                close_price = self._quotation_to_price(candle.close, instrument.price_precision)
                volume_qty = Quantity(candle.volume * instrument.lot_size, precision=instrument.size_precision)

                ts_event = int(candle.time.seconds * 1_000_000_000 + candle.time.nanos)
                ts_init = self._clock.timestamp_ns() # Or use a timestamp from the response if available

                bar = Bar(
                    bar_type=bar_type,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume_qty,
                    ts_event=ts_event,
                    ts_init=ts_init,
                )
                bars.append(bar)

            # Send the historical data response
            # self._handle_bars(bar_type, bars, correlation_id) # Check correct method name in Nautilus docs
            self._send_historical_data_response(correlation_id, bars) # Assuming this is the correct method

        except RequestError as e:
            self._log.error(f"Error requesting historical bars for {bar_type}: {e}")
            # Send error response
            # self._send_request_error_response(correlation_id, f"Error requesting bars: {e}") # Check correct method
        except Exception as e:
             self._log.exception(f"Unexpected error requesting historical bars for {bar_type}: {e}")
             # Send error response
             # self._send_request_error_response(correlation_id, f"Unexpected error requesting bars: {e}")

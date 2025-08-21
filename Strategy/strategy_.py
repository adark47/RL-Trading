from decimal import Decimal
import pandas as pd
#import fireducks.pandas as pd
#from loguru import logger
from tabulate import tabulate
from datetime import datetime, timedelta  # Дата и время, временной интервал
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators.average.ma_factory import MovingAverageFactory
from nautilus_trader.indicators.average.moving_average import MovingAverageType
from nautilus_trader.indicators.average.hma import HullMovingAverage
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.book import BookLevel
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.functions import order_side_to_str
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import OrderList
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.cache.config import CacheConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.core.data import Data
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.orders import LimitOrder

import talib


class MACrossStrategyConfig(StrategyConfig, frozen=True):
    instrument: Instrument
    primary_bar_type: BarType
    trade_size: Decimal
    ma_type: MovingAverageType.SIMPLE
    ma_fast_period: int
    ma_slow_period: int
    atr_period: int
    profit_in_ticks: int
    stoploss_in_ticks: int



class MACrossStrategy(Strategy):
    def __init__(self, config: MACrossStrategyConfig):
        super().__init__(config)

        self.dfBars = pd.DataFrame()
        self.dfBars = pd.DataFrame(columns=['time', 'open', 'close', 'high', 'low', 'volume'])
        self.dfBars.set_index('time', inplace=True)
        self.dfBars.index = pd.to_datetime(self.dfBars.index)
        self.dfBars.index.name = 'time'

        self.dfTradeTick = pd.DataFrame()
        self.dfTradeTick = pd.DataFrame(columns=['time',
                                              #'ts_event',
                                              #'instrument_id',
                                              'aggressor_side',
                                              'size',
                                              'price',
                                              ])
        self.dfTradeTick.set_index('time', inplace=True)
        self.dfTradeTick.index = pd.to_datetime(self.dfTradeTick.index)
        self.dfTradeTick.index.name = 'time'

        self.df_poc_5m = pd.DataFrame(columns=['time',
                                               'sum',
                                               'count',
                                               'price',
                                               ])
        self.df_poc_5m.set_index('time', inplace=True)
        self.df_poc_5m.index = pd.to_datetime(self.dfTradeTick.index)
        self.df_poc_5m.index.name = 'time'

        self.df_poc_15m = pd.DataFrame(columns=['time',
                                               'sum',
                                               'count',
                                               'price',
                                               ])
        self.df_poc_15m.set_index('time', inplace=True)
        self.df_poc_15m.index = pd.to_datetime(self.dfTradeTick.index)
        self.df_poc_15m.index.name = 'time'

        self.df_poc_1h = pd.DataFrame(columns=['time',
                                               'sum',
                                               'count',
                                               'price',
                                               ])
        self.df_poc_1h.set_index('time', inplace=True)
        self.df_poc_1h.index = pd.to_datetime(self.dfTradeTick.index)
        self.df_poc_1h.index.name = 'time'


        self.dfOrderBookAsks = pd.DataFrame(columns=[
                                            'price',
                                            'size',
                                            'exposure',
                                        ])
        self.dfOrderBookAsks.set_index('price', inplace=True)

        self.dfOrderBookBids = pd.DataFrame(columns=[
                                            'price',
                                            'size',
                                            'exposure',
                                        ])
        self.dfOrderBookBids.set_index('price', inplace=True)

        self.OrderBookBestBidPrice = None
        self.OrderBookBestAskPrice = None
        self.df_OrderBook = pd.DataFrame(columns=['time',
                                                  'best_bid_price',
                                                  'best_ask_price',
                                                  'max_bid_size',
                                                  'max_bid_price',
                                                  'max_ask_size',
                                                  'max_ask_price',
                                                  'sum_bid_sise',
                                                  'sum_bid_size'
                                                 ])
        self.df_OrderBook.set_index('time', inplace=True)
        self.df_OrderBook.index = pd.to_datetime(self.df_OrderBook.index)
        self.df_OrderBook.index.name = 'time'


        # Basic checks if configuration makes sense for the strategy
        PyCondition.is_true(
            config.ma_fast_period < config.ma_slow_period,
            "Invalid configuration: Fast MA period {config.ma_fast_period=} must be smaller than slow MA period {config.ma_slow_period=}",
        )

        # Create indicators
        self.ma_fast = MovingAverageFactory.create(period=config.ma_fast_period, ma_type=config.ma_type)
        self.ma_slow = MovingAverageFactory.create(period=config.ma_slow_period, ma_type=config.ma_type)

        self.hma_fast = HullMovingAverage(period=20)
        self.hma_slow = HullMovingAverage(period=50)

        self.atr = AverageTrueRange(config.atr_period)
        self.macd = MovingAverageConvergenceDivergence(12, 26)
        self.rsi = RelativeStrengthIndex(14)
        #self.trendStrength = (self.ma_fast - self.ma_slow) / self.ma_slow * 100


    def on_start(self):
        # Connect indicators with bar-type for automatic updating
        #print(self.config.instrument)
        instrument_id = self.config.instrument
        self.register_indicator_for_bars(self.config.primary_bar_type, self.ma_fast)
        self.register_indicator_for_bars(self.config.primary_bar_type, self.ma_slow)

        self.register_indicator_for_bars(self.config.primary_bar_type, self.hma_fast)
        self.register_indicator_for_bars(self.config.primary_bar_type, self.hma_slow)

        self.register_indicator_for_bars(self.config.primary_bar_type, self.atr)
        self.register_indicator_for_bars(self.config.primary_bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.primary_bar_type, self.macd)



        # Subscribe to bars
        self.subscribe_bars(self.config.primary_bar_type)


        from nautilus_trader.model.enums import BookType

        self.subscribe_order_book_at_interval(
                 instrument_id=instrument_id,
                 book_type=BookType.L2_MBP, # L3_MBO # L2_MBP # L1_MBP
                 depth=200,
                 interval_ms=500)

        self.subscribe_order_book_deltas(instrument_id=instrument_id)
        self.subscribe_quote_ticks(instrument_id=instrument_id)
        self.subscribe_trade_ticks(instrument_id=instrument_id)

        #self.subscribe_instrument_status(instrument_id=instrument_id)

        from nautilus_trader.model.data import DataType
        from nautilus_trader.model.data import InstrumentStatus

        #status_data_type = DataType(
        #         type=InstrumentStatus,
        #         metadata={"instrument_id": instrument_id},
        #)
        #self.request_data(status_data_type)

        self.request_bars(BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"))
        self.request_quote_ticks(instrument_id=instrument_id)
        self.request_trade_ticks(instrument_id=instrument_id)
        #self.request_order_book_snapshot(instrument_id=instrument_id)


            # Imbalance
        #from nautilus_trader.adapters.databento import DatabentoImbalance

        #metadata = {"instrument_id": instrument_id}
        #self.request_data(data_type=DataType(type=DatabentoImbalance, metadata=metadata))

            # Statistics
        #from nautilus_trader.adapters.databento import DatabentoStatistics

        #metadata = {"instrument_id": instrument_id}
        #self.subscribe_data(data_type=DataType(type=DatabentoStatistics, metadata=metadata))
#        self.request_data(data_type=DataType(type=DatabentoStatistics, metadata=metadata))

    def on_bar(self, bar: Bar):
        self.log.info(f"Bar: {repr(bar)}", color=LogColor.GREEN)

        ohlc = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        self.dfBars.loc[pd.to_datetime(pd.to_datetime(datetime.fromtimestamp(bar.ts_init / 1e9).strftime('%Y-%m-%dT%H:%M')))] = [
                                                                             bar.open,
                                                                             bar.close,
                                                                             bar.high,
                                                                             bar.low,
                                                                             bar.volume
                                                                             ]
        self.dfBars = self.dfBars[(datetime.today() - timedelta(days=2)):]  # делаем глубину датафрейма

        self.log.info(f"\n {tabulate(self.dfBars.tail(100), showindex=True, headers='keys', tablefmt='psql')}", color=LogColor.GREEN)
        self.log.info(f'Bar info: \n {self.dfBars.info()}', LogColor.RED)
        self.log.info(f'Bar Size: {self.dfBars.memory_usage().sum() / 1024 / 1024:.3f} MB', LogColor.RED)
        #print(tabulate(self.dfBarself.dfBars.to_csv('dfBars_1m.csv', index=True)s.resample('5min').apply(ohlc).dropna().tail(10), showindex=True, headers='keys', tablefmt='psql'))

        #self.log.info(f'Trend Strength: {self.trendStrength}%', LogColor.RED)


        #print(self.dfBars.resample('1h').apply(ohlc).dropna().tail(5))

        #if len(self.dfBars) > 5: print(detect_triangle_pattern(self.dfBars))


        # Wait until all registered indicators are initialized
        if not self.indicators_initialized():
            count_of_bars = self.cache.bar_count(self.config.primary_bar_type)
            self.log.info(
                f"Waiting for indicators to warm initialize. | Bars count {count_of_bars}",
                color=LogColor.BLUE,
            )
            return

        #print(f'ma_slow: {self.ma_slow.value}')
        #print(f'ma_fast: {self.ma_fast.value}')
        #print(f'hma_slow: {self.hma_slow.value}')
        #print(f'hma_fast: {self.hma_fast.value}')
        #print(f'atr: {self.atr.value}')
        #print(f'macd: {self.macd.value}')
        #print(f'rsi: {self.rsi.value}')
        #print((self.hma_fast.value - self.hma_slow.value) / self.hma_slow.value * 100)

        #print(talib.ATR(high=self.dfBars.high, low=self.dfBars.low, close=self.dfBars.close, timeperiod=14))
        #print(f'Bar STD close: {self.dfBars.iloc[-4:-1]["close"].std()}')
        #print(f'Bar STD volume: {self.dfBars.iloc[-4:-1]["volume"].std()}')

        print()
        print("-" * 27)
        print('5min')
        for time, group in self.dfTradeTick.resample('5min'):
            grp = group.groupby('price')['size'].agg(['sum', 'count']).sort_values(by='sum', ascending=False).head(1)
            grp['time'] = time
            grp['price'] = grp.index
            grp.set_index('time', inplace=True)
            print(grp)
            #self.df_poc_5m = pd.concat([self.df_poc_5m, grp]).sort_values(by='sum', ascending=False).drop_duplicates(subset='price', keep='first')
            self.df_poc_5m.loc[time] = [grp.iloc[-1]['sum'],
                                        grp.iloc[-1]['count'],
                                        grp.iloc[-1]['price']
                                        ]
        #self.df_poc_5m = self.df_poc_5m[(datetime.today() - timedelta(minutes=60)):]  # делаем глубину датафрейма
        print(tabulate(self.df_poc_5m, showindex=True, headers='keys', tablefmt='psql'))


        print("-" * 27)
        print('15min')
        for time, group in self.dfTradeTick.resample('15min'):
            grp = group.groupby('price')['size'].agg(['sum', 'count']).sort_values(by='sum', ascending=False).head(1)
            grp['time'] = time
            grp['price'] = grp.index
            grp.set_index('time', inplace=True)
            print(grp)
            # self.df_poc_5m = pd.concat([self.df_poc_5m, grp]).sort_values(by='sum', ascending=False).drop_duplicates(subset='price', keep='first')
            self.df_poc_15m.loc[time] = [grp.iloc[-1]['sum'],
                                        grp.iloc[-1]['count'],
                                        grp.iloc[-1]['price']
                                        ]
        #self.df_poc_15m = self.df_poc_15m[(datetime.today() - timedelta(hours=8)):]  # делаем глубину датафрейма
        print(tabulate(self.df_poc_15m, showindex=True, headers='keys', tablefmt='psql'))


        print("-" * 27)
        print('1H')
        for time, group in self.dfTradeTick.resample('1h'):
            grp = group.groupby('price')['size'].agg(['sum', 'count']).sort_values(by='sum', ascending=False).head(1)
            grp['time'] = time
            grp['price'] = grp.index
            grp.set_index('time', inplace=True)
            print(grp)
            # self.df_poc_5m = pd.concat([self.df_poc_5m, grp]).sort_values(by='sum', ascending=False).drop_duplicates(subset='price', keep='first')
            self.df_poc_1h.loc[time] = [grp.iloc[-1]['sum'],
                                        grp.iloc[-1]['count'],
                                        grp.iloc[-1]['price']
                                        ]
        #self.df_poc_1h = self.df_poc_1h[(datetime.today() - timedelta(hours=12)):]  # делаем глубину датафрейма
        print(tabulate(self.df_poc_1h, showindex=True, headers='keys', tablefmt='psql'))







        # Note: If we got here, all registered indicator are initialized

        # BUY LOGIC
#        if self.ma_fast.value > self.ma_slow.value:  # If fast EMA is above slow EMA
#            if self.portfolio.is_flat(self.config.instrument.id):  # If we are flat
#                self.cancel_all_orders(
#                    self.config.instrument.id
#                )  # Make sure all waiting orders are cancelled
#                self.fire_trade(OrderSide.BUY, bar)  # Fire buy order
#            if self.portfolio.is_net_short(self.config.instrument.id):  # We are short already
#                self.cancel_all_orders(
#                    self.config.instrument.id
#                )  # Make sure all waiting orders are cancelled
#                self.close_all_positions(self.config.instrument.id)  # Let's close current position
#                self.fire_trade(OrderSide.BUY, bar)  # Fire buy order

        # SELL LOGIC
#        if self.ma_fast.value < self.ma_slow.value:
#            if self.portfolio.is_flat(self.config.instrument.id):
#                self.cancel_all_orders(self.config.instrument.id)
#                self.fire_trade(OrderSide.SELL, bar)
#            if self.portfolio.is_net_long(self.config.instrument.id):
#                self.cancel_all_orders(self.config.instrument.id)
#                self.close_all_positions(self.config.instrument.id)
#                self.fire_trade(OrderSide.BUY, bar)

    def on_order_book(self, order_book: OrderBook) -> None:     # Represents the best bid and ask prices along with their sizes at the top-of-book.
        """
        Actions to be performed when the strategy is running and receives an order book.

        Parameters
        ----------
        order_book : OrderBook
            The order book received.

        """
        # For debugging (must add a subscription)
        #self.log.info(f"\n{order_book.instrument_id}\n{order_book.pprint(8)}", LogColor.CYAN)
        #print(order_book.asks())
        #print(order_book.bids)

        self.dfOrderBookAsks = pd.DataFrame(columns=[
                                            'price',
                                            'size',
                                            'exposure',
                                        ])
        self.dfOrderBookAsks.set_index('price', inplace=True)
        for ask in order_book.asks():
            self.dfOrderBookAsks.loc[ask.price] = [ask.size(), ask.exposure()]

        self.dfOrderBookBids = pd.DataFrame(columns=[
                                            'price',
                                            'size',
                                            'exposure',
                                        ])
        self.dfOrderBookBids.set_index('price', inplace=True)
        for bid in order_book.bids():
            self.dfOrderBookBids.loc[bid.price] = [bid.size(), bid.exposure()]


        #self.dfOrderBookAsks = pd.DataFrame.from_dict(order_book.asks())
        #self.dfOrderBookBids = pd.DataFrame.from_dict(order_book.bids())

        print(order_book.best_bid_price())
        print(order_book.best_ask_price())
        self.OrderBookBestBidPrice = order_book.best_bid_price()
        self.OrderBookBestAskPrice = order_book.best_ask_price()





        self.df_OrderBook.loc[pd.to_datetime(datetime.fromtimestamp(order_book.ts_init / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%f'))] = [
                            order_book.best_bid_price(),                                                    # 'best bid price'
                            order_book.best_ask_price(),                                                    # 'best ask price'
                            self.dfOrderBookBids.sort_values(by='size', ascending=True).iloc[-1]['size'],   # 'max bid size'
                            self.dfOrderBookBids.sort_values(by='size', ascending=True).index.values[-1],   # 'max bid price'
                            self.dfOrderBookAsks.sort_values(by='size', ascending=True).iloc[-1]['size'],   # 'max ask size'
                            self.dfOrderBookAsks.sort_values(by='size', ascending=True).index.values[-1],   # 'max ask price'
                            self.dfOrderBookAsks['size'].sum(),                                             # 'sum bid sise'
                            self.dfOrderBookBids['size'].sum()                                              # 'sum ask size'
                            ]
        self.df_OrderBook = self.df_OrderBook[(datetime.today() - timedelta(minutes=480)):]  # делаем глубину датафрейма
        print(tabulate(self.df_OrderBook.tail(5), showindex=True, headers='keys', tablefmt='psql'))



        #print(datetime.fromtimestamp(order_book.ts_init / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%f'))
        #self.log.info(f"OrderBook Best Bid Price: {self.OrderBookBestBidPrice}", LogColor.CYAN)
        #self.log.info(f"OrderBook Best Ask Price: {self.OrderBookBestAskPrice}", LogColor.CYAN)

#        print(order_book.spread())



        #self.log.info(repr(order_book), LogColor.CYAN)



#        self.log.info(f"OrderBook Asks: \n {self.dfOrderBookAsks['size'].sum()}", LogColor.CYAN)
#        self.log.info(f"OrderBook Asks: \n {self.dfOrderBookBids['size'].sum()}", LogColor.CYAN)
        #self.log.info(f"OrderBook Bids: \n {self.dfOrderBookAsks.sort_values(by='size', ascending=False).head(3)}", LogColor.CYAN)
        #print(order_book.book_type)
        #print(order_book)


        #print(self.dfOrderBookBids.sort_values(by='size', ascending=False).head(3))

        # Если дано числа A и B, такие что A>B и необходимо узнать на сколько процентов число A больше числа B:  P = (A - B) / B ·100 %

#        for bid in self.dfOrderBookBids.sort_values(by='size', ascending=False).head(3).itertuples():
#            print(bid.Index, order_book.best_bid_price())

#            print(((order_book.best_bid_price() - bid.Index) / bid.Index * 100))

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        """
        Actions to be performed when the strategy is running and receives order book
        deltas.

        Parameters
        ----------
        deltas : OrderBookDeltas
            The order book deltas received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(deltas), LogColor.CYAN)

        book = self.cache.order_book(deltas.instrument_id)

        #print(book.asks())
        #print(book.best_bid_price)
        #print(book.best_ask_price)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """
        Actions to be performed when the strategy is running and receives a quote tick.

        Parameters
        ----------
        tick : QuoteTick
            The tick received.

        """
        # For debugging (must add a subscription)
        #self.log.info(repr(tick), LogColor.MAGENTA)
        pass

    def on_trade_tick(self, tick: TradeTick) -> None:   # A single trade/match event between counterparties.
        """
        Actions to be performed when the strategy is running and receives a trade tick.

        Parameters
        ----------
        tick : TradeTick
            The tick received.

        """
        #print(dt.strftime('%Y-%m-%dT%H:%M:%S.%f'))

        self.dfTradeTick.loc[pd.to_datetime(datetime.fromtimestamp(tick.ts_init / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%f'))] = [
                                            #datetime.fromtimestamp(tick.ts_event / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%f'),
                                            #tick.instrument_id,
                                            tick.aggressor_side,
                                            tick.size,
                                            tick.price,
                                            ]
        #self.dfTradeTick = self.dfTradeTick[(datetime.today() - timedelta(minutes=480)):]  # делаем глубину датафрейма


#        self.log.info(f'dfTradeTick: {self.dfTradeTick.memory_usage().sum() / 1024 / 1024:.3f} MB', LogColor.RED)
#        self.log.info(f"\n{tabulate(self.dfTradeTick.tail(5), showindex=True, headers='keys', tablefmt='psql')}", LogColor.MAGENTA)
#        self.log.info(f'dfTradeTick: \n{self.dfTradeTick.info()}', LogColor.RED)

    def fire_trade(self, order_side: OrderSide, last_bar: Bar):
        last_price = last_bar.close

        # Prepare profit/stoploss prices
        if order_side == OrderSide.BUY:
            profit_price = (
                last_price + self.config.profit_in_ticks * self.config.instrument.price_increment
            )
            stoploss_price = (
                last_price - self.config.stoploss_in_ticks * self.config.instrument.price_increment
            )
        elif order_side == OrderSide.SELL:
            profit_price = (
                last_price - self.config.profit_in_ticks * self.config.instrument.price_increment
            )
            stoploss_price = (
                last_price + self.config.stoploss_in_ticks * self.config.instrument.price_increment
            )
        else:
            raise ValueError(f"Order side: {order_side} is not supported.")

        # Prepare bracket order (bracket order is entry order with related contingent profit / stoploss orders)
        bracket_order_list: OrderList = self.order_factory.bracket(
            instrument_id=self.config.instrument.id,
            order_side=order_side,
            quantity=self.config.instrument.make_qty(self.config.trade_size),
            entry_order_type=OrderType.MARKET,  # enter trade with MARKET order
            sl_trigger_price=self.config.instrument.make_price(
                stoploss_price
            ),  # stoploss is always MARKET order (fixed in Nautilus)
            tp_order_type=OrderType.LIMIT,  # profit is LIMIT order
            tp_price=self.config.instrument.make_price(
                profit_price
            ),  # set price for profit LIMIT order
        )

        # Log order
        self.log.info(
            f"Order: {order_side_to_str(order_side)} | Last price: {last_price} | Profit: {profit_price} | Stoploss: {stoploss_price}",
            color=LogColor.BLUE,
        )

        # Submit order
        self.submit_order_list(bracket_order_list)

    def on_stop(self):
        pass

    def on_event(self, event: Event) -> None:
        """
        Actions to be performed when the strategy is running and receives an event.

        Parameters
        ----------
        event : Event
            The event received.

        """
        # Optionally implement

        print(event)


    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        #import os
        #os.mkdir('data')

        ohlc = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Optionally implement
        #self.dfBars.to_csv('./data/dfBars_1m.csv', index=True)
        #self.dfBars.resample('5min').apply(ohlc).dropna().to_csv('./data/dfBars_5m.csv', index=True)
        #self.dfBars.resample('15min').apply(ohlc).dropna().to_csv('./data/dfBars_15m.csv', index=True)
        #self.dfBars.resample('1h').apply(ohlc).dropna().to_csv('./data/dfBars_1h.csv', index=True)
        #self.dfTradeTick.to_csv('./data/dfTradeTick.csv', index=True)
        #self.df_OrderBook.to_csv('./data/df_OrderBook.csv', index=True)
        #self.df_poc_5m.to_csv('./data/df_poc_5m.csv', index=True)
        #self.df_poc_15m.to_csv('./data/df_poc_15m.csv', index=True)
        #self.df_poc_1h.to_csv('./data/df_poc_1h.csv', index=True)
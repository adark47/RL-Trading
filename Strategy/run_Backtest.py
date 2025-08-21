# run_Backtest.py

import settings

from decimal import Decimal

from strategy import Strategy
from strategy import StrategyConfig

from data_provider import prepare_data_1min
from nautilus_trader.indicators.average.moving_average import MovingAverageType
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model import Bar
from nautilus_trader.model import TraderId
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.objects import Money


if __name__ == "__main__":

    # ----------------------------------------------------------------------------------
    # 1. Configure and create backtest engine
    # ----------------------------------------------------------------------------------

    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST-SIGNALS-001"),  # Unique identifier for this backtest
        logging=LoggingConfig(log_level="INFO"),
    )
    engine = BacktestEngine(config=engine_config)

    # ----------------------------------------------------------------------------------
    # 2. Prepare market data
    # ----------------------------------------------------------------------------------

    prepared_data: dict = prepare_data_1min()
    venue_name: str = prepared_data["venue_name"]
    _instrument: Instrument = prepared_data["instrument"]
    _1min_bartype = prepared_data["bar_type"]
    _1min_bars: list[Bar] = prepared_data["bars_list"]

    # ----------------------------------------------------------------------------------
    # 3. Configure trading environment
    # ----------------------------------------------------------------------------------

    # Set up the trading venue with a margin account
    engine.add_venue(
        venue=Venue(venue_name),
        oms_type=OmsType.NETTING,  # Use a netting order management system
        account_type=AccountType.MARGIN,  # Use a margin trading account
        starting_balances=[Money(1_000_000, USD)],  # Set initial capital
        base_currency=USD,  # Account currency
        default_leverage=Decimal(1),  # No leverage (1:1)
    )

    # Register the trading instrument
    engine.add_instrument(_instrument)

    # Load historical market data
    engine.add_data(_1min_bars)

    # ----------------------------------------------------------------------------------
    # 4. Configure and run strategy
    # ----------------------------------------------------------------------------------

    # Create strategy configuration with proper warmup_bars
    strategy_config = StrategyConfig(
        instrument_id=_instrument.id,  # Используем instrument.id вместо всего объекта instrument
        primary_bar_type=_1min_bartype,
        trade_size=Decimal(settings.trade_size),  # taken from settings
        warmup_bars=35  # Минимальное количество баров для расчета MACD(12,26,9)
    )

    # Create and register the strategy
    strategy = Strategy(config=strategy_config)
    engine.add_strategy(strategy)

    # Execute the backtest
    engine.run()

    # Clean up resources
    engine.dispose()
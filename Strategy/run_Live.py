# run_Live.py

import settings

from nautilus_trader.adapters.bybit.common.enums import BybitProductType
from nautilus_trader.adapters.bybit.config import (
    BybitDataClientConfig,
    BybitExecClientConfig,
)
from nautilus_trader.adapters.bybit.factories import (
    BybitLiveDataClientFactory,
    BybitLiveExecClientFactory,
)
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveExecEngineConfig,
    TradingNodeConfig,
    LoggingConfig,
)
from nautilus_trader.indicators.average.moving_average import MovingAverageType
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model import TraderId
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.backtest.engine import Decimal

from strategy import Strategy, StrategyConfig


def _validate_credentials() -> None:
    """Ensure BYBIT API credentials are available before starting the node."""
    if not settings.api_key or not settings.api_secret:
        raise ValueError("BYBIT API credentials are not set in settings (api_key / api_secret).")


def build_node() -> TradingNode:
    """Create, configure and build a TradingNode instance ready to run."""

    _validate_credentials()

    # SPOT/LINEAR product type and symbol
    product_type = BybitProductType.LINEAR
    symbol = f"{settings.symbol}-{product_type.value.upper()}"

        # Strategy configuration -------------------------------------------------
    strat_config = StrategyConfig(
        instrument=InstrumentId.from_str(f"{symbol}.BYBIT"),
        primary_bar_type=BarType.from_str(f"{symbol}.BYBIT-1-MINUTE-LAST-EXTERNAL"),
        trade_size=Decimal(settings.trade_size),  # taken from settings
    )

    instrument_provider_cfg = InstrumentProviderConfig(load_all=True)

    # Node configuration -----------------------------------------------------
    node_config = TradingNodeConfig(
        trader_id=TraderId(settings.trader_id_live),
        logging=LoggingConfig(log_level="INFO", use_pyo3=True),
        exec_engine=LiveExecEngineConfig(
            reconciliation=True,
            reconciliation_lookback_mins=1440,
        ),
        data_clients={
            "BYBIT": BybitDataClientConfig(
                    api_key=settings.api_key,
                    api_secret=settings.api_secret,
                    base_url_http=None,
                    instrument_provider=instrument_provider_cfg,
                product_types=[product_type],
                    testnet=False,
            ),
        },
        exec_clients={
            "BYBIT": BybitExecClientConfig(
                    api_key=settings.api_key,
                    api_secret=settings.api_secret,
                    base_url_http=None,
                    base_url_ws_private=None,
                    instrument_provider=instrument_provider_cfg,
                    product_types=[product_type],
                    testnet=False,
                    max_retries=5,

            ),
        },
        timeout_connection=settings.timeout_connection,
        timeout_reconciliation=settings.timeout_reconciliation,
        timeout_portfolio=settings.timeout_portfolio,
        timeout_disconnection=settings.timeout_disconnection,
        timeout_post_stop=settings.timeout_post_stop,
    )

        # Build and return the node ---------------------------------------------
    node = TradingNode(config=node_config)
    node.trader.add_strategy(Strategy(config=strat_config))
    node.add_data_client_factory("BYBIT", BybitLiveDataClientFactory)
    node.add_exec_client_factory("BYBIT", BybitLiveExecClientFactory)
    node.build()
    return node


def main() -> None:
    """Entry point for running the trading node."""
    node = build_node()
    try:
        node.run()
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C
        pass
    finally:
        node.stop()
        node.dispose()


if __name__ == "__main__":
    main()


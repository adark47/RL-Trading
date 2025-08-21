"""
Tinkoff data schemas.
"""

from dataclasses import dataclass
from typing import List, Optional

from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


@dataclass
class TinkoffInstrument:
    """
    Tinkoff instrument.
    """
    figi: str
    ticker: str
    class_code: str
    isin: str
    lot: int
    currency: str
    min_price_increment: float
    min_price_increment_amount: float
    short_enabled_flag: bool
    name: str
    exchange: str
    instrument_type: str
    point_value: Optional[float] = None
    futures_margin: Optional[float] = None
    asset_type: Optional[str] = None
    asset_size: Optional[float] = None
    basic_asset: Optional[str] = None
    basic_asset_size: Optional[float] = None
    country_of_risk: Optional[str] = None
    country_of_risk_name: Optional[str] = None
    sector: Optional[str] = None
    issue_size: Optional[int] = None
    issue_size_plan: Optional[int] = None
    nominal: Optional[float] = None
    trading_status: Optional[str] = None
    otc_flag: Optional[bool] = None
    buy_available_flag: Optional[bool] = None
    sell_available_flag: Optional[bool] = None
    iso_currency_name: Optional[str] = None
    min_quantity: Optional[int] = None
    uid: Optional[str] = None
    real_exchange: Optional[str] = None
    position_uid: Optional[str] = None
    dshort: Optional[float] = None
    dshort_min: Optional[float] = None
    dlong: Optional[float] = None
    dlong_min: Optional[float] = None
    short_enabled_flag: Optional[bool] = None
    kshort: Optional[float] = None
    klong: Optional[float] = None
    futures_type: Optional[str] = None
    asset_currency: Optional[str] = None
    futures_direction: Optional[str] = None
    first_trade_date: Optional[str] = None
    last_trade_date: Optional[str] = None
    futures_margin: Optional[float] = None
    nominal_order_unit: Optional[str] = None
    trading_status: Optional[str] = None


@dataclass
class TinkoffBar:
    """
    Tinkoff candle.
    """
    figi: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    time: str
    is_complete: bool


@dataclass
class TinkoffOrderBook:
    """
    Tinkoff order book.
    """
    figi: str
    depth: int
    bids: List[tuple[float, int]]
    asks: List[tuple[float, int]]
    time: str
    is_consistent: bool


@dataclass
class TinkoffTrade:
    """
    Tinkoff trade.
    """
    figi: str
    direction: str
    price: float
    quantity: int
    time: str
    id: str
    exchange_trade_id: str
    exchange_trade_time: str


@dataclass
class TinkoffOrder:
    """
    Tinkoff order.
    """
    order_id: str
    figi: str
    direction: str
    status: str
    requested_lots: int
    executed_lots: int
    type: str
    price: float
    stop_price: Optional[float] = None
    expiration_time: Optional[str] = None
    create_time: str
    update_time: str
    instrument_uid: Optional[str] = None
    order_type: Optional[str] = None
    account_id: Optional[str] = None
    operation: Optional[str] = None
    message: Optional[str] = None
    reject_reason: Optional[str] = None
    initial_security_price: Optional[float] = None
    steps: Optional[int] = None
    initial_order_price: Optional[float] = None
    initial_commission: Optional[float] = None
    executed_commission: Optional[float] = None
    aci_value: Optional[float] = None
    figi_type: Optional[str] = None
    instrument_uid: Optional[str] = None
    order_request_id: Optional[str] = None
    is_margin_call: Optional[bool] = None
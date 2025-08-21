"""
Tinkoff utility functions.
"""

from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from nautilus_trader.adapters.tinkoff.common import TINKOFF_VENUE
from nautilus_trader.adapters.tinkoff.schemas import TinkoffInstrument


def tinkoff_instrument_to_nautilus(tinkoff_instrument: TinkoffInstrument):
    """
    Convert a Tinkoff instrument to a Nautilus instrument.
    """
    from nautilus_trader.model.instruments.equity import Equity
    from nautilus_trader.model.instruments.futures_contract import FuturesContract
    from nautilus_trader.model.instruments.option_contract import OptionContract
    from nautilus_trader.model.instruments.currency_pair import CurrencyPair

    # Определяем тип инструмента
    instrument_type = tinkoff_instrument.instrument_type.lower()

    # Создаем идентификатор инструмента
    instrument_id = InstrumentId(
        symbol=tinkoff_instrument.ticker,
        venue=TINKOFF_VENUE
    )

    # Создаем символ с кодом класса
    raw_symbol = f"{tinkoff_instrument.class_code}.{tinkoff_instrument.ticker}"

    # Определяем базовую валюту
    base_currency = tinkoff_instrument.currency.upper()

    # Определяем точность цены
    price_precision = len(str(tinkoff_instrument.min_price_increment).split('.')[1]) if '.' in str(
        tinkoff_instrument.min_price_increment) else 0

    # Определяем точность количества
    quantity_precision = len(str(tinkoff_instrument.lot).split('.')[1]) if '.' in str(tinkoff_instrument.lot) else 0

    if instrument_type == "share":
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=raw_symbol,
            currency=base_currency,
            price_precision=price_precision,
            size_precision=quantity_precision,
            price_increment=Price(tinkoff_instrument.min_price_increment, price_precision),
            size_increment=Quantity(1, quantity_precision),
            lot_size=Quantity(tinkoff_instrument.lot, quantity_precision),
            isin=tinkoff_instrument.isin,
            margin_init=0.0,  # Для акций обычно 0
            max_quantity=None,
            min_quantity=Quantity(1, quantity_precision),
            max_notional=None,
            min_notional=None,
            max_price=None,
            min_price=None,
            max_quantity=Quantity(1000000, quantity_precision),  # Условное значение
            min_quantity=Quantity(1, quantity_precision),
            margin_maintenance=0.0,
            maker_fee=0.0005,  # Примерный размер комиссии
            taker_fee=0.0005,
        )

    elif instrument_type == "bond":
        # Для облигаций номинал обычно 1000
        nominal = tinkoff_instrument.nominal or 1000
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=raw_symbol,
            currency=base_currency,
            price_precision=price_precision,
            size_precision=quantity_precision,
            price_increment=Price(tinkoff_instrument.min_price_increment, price_precision),
            size_increment=Quantity(1, quantity_precision),
            lot_size=Quantity(tinkoff_instrument.lot, quantity_precision),
            isin=tinkoff_instrument.isin,
            margin_init=0.0,
            max_quantity=None,
            min_quantity=Quantity(1, quantity_precision),
            max_notional=None,
            min_notional=None,
            max_price=None,
            min_price=None,
            max_quantity=Quantity(1000000, quantity_precision),
            min_quantity=Quantity(1, quantity_precision),
            margin_maintenance=0.0,
            maker_fee=0.0005,
            taker_fee=0.0005,
        )

    elif instrument_type == "future":
        # Для фьючерсов используем точку как единицу изменения цены
        point_value = tinkoff_instrument.point_value or 1
        return Future(
            instrument_id=instrument_id,
            raw_symbol=raw_symbol,
            underlying=tinkoff_instrument.basic_asset,
            currency=base_currency,
            price_precision=price_precision,
            size_precision=quantity_precision,
            price_increment=Price(tinkoff_instrument.min_price_increment, price_precision),
            size_increment=Quantity(1, quantity_precision),
            lot_size=Quantity(tinkoff_instrument.lot, quantity_precision),
            margin_init=0.1,  # Примерная начальная маржа
            margin_maintenance=0.05,
            max_quantity=None,
            min_quantity=Quantity(1, quantity_precision),
            max_notional=None,
            min_notional=None,
            max_price=None,
            min_price=None,
            max_quantity=Quantity(1000, quantity_precision),
            min_quantity=Quantity(1, quantity_precision),
            maker_fee=0.0002,
            taker_fee=0.0002,
            contract_value=float(point_value),
        )

    elif instrument_type == "currency":
        # Для валютных пар
        quote_currency = "USD" if base_currency == "RUB" else "RUB"
        return CurrencyPair(
            instrument_id=instrument_id,
            raw_symbol=raw_symbol,
            base_currency=base_currency,
            quote_currency=quote_currency,
            price_precision=price_precision,
            size_precision=quantity_precision,
            price_increment=Price(tinkoff_instrument.min_price_increment, price_precision),
            size_increment=Quantity(1, quantity_precision),
            lot_size=Quantity(tinkoff_instrument.lot, quantity_precision),
            margin_init=0.05,
            margin_maintenance=0.03,
            max_quantity=None,
            min_quantity=Quantity(1, quantity_precision),
            max_notional=None,
            min_notional=None,
            max_price=None,
            min_price=None,
            max_quantity=Quantity(1000000, quantity_precision),
            min_quantity=Quantity(1, quantity_precision),
            maker_fee=0.00005,
            taker_fee=0.00005,
        )

    else:
        # По умолчанию используем базовый класс
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=raw_symbol,
            currency=base_currency,
            price_precision=price_precision,
            size_precision=quantity_precision,
            price_increment=Price(tinkoff_instrument.min_price_increment, price_precision),
            size_increment=Quantity(1, quantity_precision),
            lot_size=Quantity(tinkoff_instrument.lot, quantity_precision),
            isin=tinkoff_instrument.isin,
            margin_init=0.0,
            max_quantity=None,
            min_quantity=Quantity(1, quantity_precision),
            max_notional=None,
            min_notional=None,
            max_price=None,
            min_price=None,
            max_quantity=Quantity(1000000, quantity_precision),
            min_quantity=Quantity(1, quantity_precision),
            margin_maintenance=0.0,
            maker_fee=0.0005,
            taker_fee=0.0005,
        )
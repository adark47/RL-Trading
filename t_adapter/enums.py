# t_adapter/enums.py

# -*- coding: utf-8 -*-
"""
Tinkoff Invest adapter specific enums.
"""

from enum import Enum

class TinkoffProductType(Enum):
    """
    Represents a Tinkoff Invest product type.
    """
    STOCK = "stock"
    FUTURE = "future"
    OPTION = "option"
    CURRENCY = "currency" # Forex CFDs etc.
    ETF = "etf"
    BOND = "bond"
    # Add other relevant types as needed from Tinkoff Invest API

class TinkoffInstrumentType(Enum):
    """
    Represents the type of instrument as defined by Tinkoff Invest API.
    This might be used for mapping or filtering.
    """
    SHARE = "share"
    FUTURE = "future"
    OPTION = "option"
    CURRENCY = "currency"
    ETF = "etf"
    BOND = "bond"
    # Add other types from Tinkoff Invest API InstrumentType enum

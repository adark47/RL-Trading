# t_adapter/common.py

# -*- coding: utf-8 -*-
"""
Tinkoff Invest adapter common utilities and constants.
"""

from nautilus_trader.model.identifiers import Venue

# Define the venue ID for Tinkoff Invest
TINKOFF_VENUE: Venue = Venue("TINKOFF")

# Map Tinkoff Invest candle intervals to Nautilus BarType specifications if needed
# This might be handled dynamically based on subscription request

# Constants for Tinkoff Invest API interaction, if any specific ones are needed beyond the client

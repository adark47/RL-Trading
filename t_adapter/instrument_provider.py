# t_adapter/instrument_provider.py

# -*- coding: utf-8 -*-
"""
Tinkoff Invest adapter instrument provider.
"""

import logging
from typing import Dict, List, Optional, Any

from tinkoff.invest import Client, RequestError
from tinkoff.invest.schemas import Share, Future, InstrumentIdType

from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument, Equity, FuturesContract
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.currency import CurrencyType

from t_adapter.common import TINKOFF_VENUE
from t_adapter.enums import TinkoffInstrumentType, TinkoffProductType
from t_adapter.config import TAdapterDataClientConfig # Might be needed for filters


class TAdapterInstrumentProvider(InstrumentProvider):
    """
    Provides instrument definitions from Tinkoff Invest.
    """

    def __init__(
        self,
        client: Client, # Tinkoff Invest gRPC client instance
        config: TAdapterDataClientConfig, # To access product_types filter and is_sandbox
        logger: logging.Logger,
    ):
        """
        Initialize a new instance of the ``TAdapterInstrumentProvider`` class.

        Parameters
        ----------
        client : Client
            The Tinkoff Invest gRPC client instance.
        config : TAdapterDataClientConfig
            The adapter configuration.
        logger : logging.Logger
            The logger for this component.
        """
        super().__init__(config=config, logger=logger)

        self._client = client
        self._is_sandbox = config.is_sandbox
        self._product_types_filter = config.product_types or []

        # Cache for instrument data from Tinkoff API
        self._instrument_data_cache: Dict[str, Any] = {} # Key: figi, Value: Tinkoff instrument proto

    async def load_all_async(self, filters: Optional[Dict] = None) -> None:
        """
        Load all instruments matching the filters into the provider.

        Parameters
        ----------
        filters : dict, optional
            Not directly used here, filtering is done via config.product_types.
            The standard Nautilus filters might be applied post-loading if needed.
        """
        self._log.info("Loading instruments from Tinkoff Invest API")
        try:
            # Load shares
            if not self._product_types_filter or TinkoffProductType.STOCK in self._product_types_filter:
                shares_response = self._client.instruments.shares()
                for share in shares_response.instruments:
                    self._instrument_data_cache[share.figi] = share
                    nautilus_instrument = self._parse_equity(share)
                    if nautilus_instrument:
                        self.add(nautilus_instrument)

            # Load futures
            if not self._product_types_filter or TinkoffProductType.FUTURE in self._product_types_filter:
                 futures_response = self._client.instruments.futures()
                 for future in futures_response.instruments:
                    self._instrument_data_cache[future.figi] = future
                    nautilus_instrument = self._parse_futures_contract(future)
                    if nautilus_instrument:
                        self.add(nautilus_instrument)

            # Load other instrument types (ETF, Bonds, etc.) as needed
            # Following the same pattern...

            self._log.info(f"Loaded {len(self._instruments)} instruments from Tinkoff Invest API")

        except RequestError as e:
            self._log.error(f"Error loading instruments from Tinkoff Invest API: {e}")

    async def load_ids_async(
        self,
        instrument_ids: List[InstrumentId],
        filters: Optional[Dict] = None,
    ) -> None:
        """
        Load specific instruments by their IDs.

        Parameters
        ----------
        instrument_ids : list[InstrumentId]
            The instrument IDs to load.
        filters : dict, optional
            Not used in this implementation.
        """
        self._log.info(f"Loading specific instruments: {instrument_ids}")
        for instrument_id in instrument_ids:
            # Assuming InstrumentId is formatted as SYMBOL.VENUE or FIGI.VENUE
            # We'll try to extract FIGI first, then fallback to symbol search
            parts = instrument_id.value.split('.')
            identifier = parts[0] if parts else instrument_id.value

            # Check if identifier is a FIGI (common format from Tinkoff)
            if identifier.startswith('BBG') and len(identifier) > 10: # Basic FIGI check
                 if identifier in self._instrument_data_cache:
                     # Already loaded
                     continue
                 try:
                     # Try getting instrument by FIGI
                     response = self._client.instruments.get_instrument_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, class_code="", id=identifier)
                     instrument_proto = response.instrument
                     self._instrument_data_cache[instrument_proto.figi] = instrument_proto

                     # Determine type and parse
                     instrument_type = TinkoffInstrumentType(instrument_proto.instrument_type.lower())
                     nautilus_instrument = None
                     if instrument_type == TinkoffInstrumentType.SHARE:
                         nautilus_instrument = self._parse_equity(instrument_proto)
                     elif instrument_type == TinkoffInstrumentType.FUTURE:
                         nautilus_instrument = self._parse_futures_contract(instrument_proto)
                     # Add other types...

                     if nautilus_instrument:
                         self.add(nautilus_instrument)
                     continue # Move to next instrument_id
                 except RequestError as e:
                     self._log.warning(f"Failed to load instrument by FIGI {identifier}: {e}")

            # If not a FIGI or FIGI lookup failed, try searching by ticker
            # This is less reliable as ticker might not be unique across markets
            # We'll need to make assumptions or require more specific identifiers
            self._log.warning(f"Could not load instrument {instrument_id} by FIGI. Loading by ticker not implemented robustly. Please use FIGI.")

    async def load_async(self, instrument_id: InstrumentId, filters: Optional[Dict] = None) -> None:
        """
        Load a single instrument by its ID.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to load.
        filters : dict, optional
            Not used in this implementation.
        """
        await self.load_ids_async([instrument_id], filters)

    def _parse_equity(self, share_proto: Share) -> Optional[Equity]:
        """
        Parse a Tinkoff Invest Share proto into a Nautilus Equity instrument.

        Parameters
        ----------
        share_proto : Share
            The Tinkoff Share instrument proto.

        Returns
        -------
        Equity or None
        """
        try:
            instrument_id = InstrumentId.from_str(f"{share_proto.figi}.{TINKOFF_VENUE.value}")
            # Nautilus expects precision for price and size. Tinkoff provides min_price_increment.
            # We need to calculate decimal places. This is a simplified approach.
            # A more robust solution would involve checking the actual lot size and min price increment more carefully.
            price_precision = len(str(share_proto.min_price_increment.nano).rstrip('0')) if share_proto.min_price_increment.nano != 0 else 0
            if share_proto.min_price_increment.units > 0:
                # Handle units part if necessary, but usually nano is sufficient for precision calc
                price_precision = max(price_precision, len(str(share_proto.min_price_increment.units * 1000000000 + share_proto.min_price_increment.nano).rstrip('0')))

            # Size precision is usually related to lot size. Lot size is integer in Tinkoff.
            size_precision = 0 # Shares are typically whole units, lots are handled by multiplier

            currency_code = share_proto.currency.upper()
            # Ensure currency is registered in Nautilus if not default
            # This might require pre-registering common currencies or fetching them dynamically
            # For now, assuming common currencies are available or will be handled by Nautilus core
            currency = Currency.from_str(currency_code)

            # Using lot as the multiplier for contract size if applicable, or 1 if standard share
            # Tinkoff lot is the number of shares in a lot. Nautilus contract_size is usually 1 for stocks.
            # We'll set it to 1 for standard shares. If lots are significant for trading, this needs adjustment.
            contract_size = 1 # Or share_proto.lot if lot trading is significant

            # Tinkoff API provides ISO currency string, Nautilus uses Currency object
            return Equity(
                instrument_id=instrument_id,
                native_symbol=share_proto.ticker,
                currency=currency,
                price_precision=price_precision,
                size_precision=size_precision,
                multiplier=1, # Usually 1 for shares
                lot_size=share_proto.lot, # Lot size as defined by exchange
                isin=share_proto.isin,
                # margin_init, margin_maint etc. would come from risk rules if available
                ts_event=0, # This should ideally be the last update timestamp from Tinkoff
                ts_init=0, # This should be the timestamp when we created this object
            )
        except Exception as e:
            self._log.error(f"Error parsing Tinkoff Share {share_proto.figi} into Nautilus Equity: {e}")
            return None

    def _parse_futures_contract(self, future_proto: Future) -> Optional[FuturesContract]:
        """
        Parse a Tinkoff Invest Future proto into a Nautilus FuturesContract instrument.

        Parameters
        ----------
        future_proto : Future
            The Tinkoff Future instrument proto.

        Returns
        -------
        FuturesContract or None
        """
        try:
            instrument_id = InstrumentId.from_str(f"{future_proto.figi}.{TINKOFF_VENUE.value}")

            # Similar precision logic as for Equity
            price_precision = len(str(future_proto.min_price_increment.nano).rstrip('0')) if future_proto.min_price_increment.nano != 0 else 0
            size_precision = 0 # Futures lots are usually handled by contract_size/multiplier

            currency_code = future_proto.currency.upper()
            currency = Currency.from_str(currency_code)

            # Contract size for futures is crucial. It's the number of units of the underlying per contract.
            # Tinkoff provides `basic_asset_size` which is the size of the basic asset (e.g., 1000 barrels for oil futures).
            # `lot` is the number of contracts in a lot.
            # `multiplier` in Nautilus for futures is often the basic_asset_size or a related value.
            # This needs careful mapping based on Tinkoff's definition. Assuming `basic_asset_size` for now.
            contract_size = future_proto.basic_asset_size

            # Expiration and activation timestamps need conversion from google.protobuf.timestamp_pb2.Timestamp
            import datetime
            expiration_ts = int(future_proto.last_trade_date.seconds * 1_000_000_000 + future_proto.last_trade_date.nanos) if future_proto.last_trade_date else 0
            activation_ts = int(future_proto.first_trade_date.seconds * 1_000_000_000 + future_proto.first_trade_date.nanos) if future_proto.first_trade_date else 0

            return FuturesContract(
                instrument_id=instrument_id,
                native_symbol=future_proto.ticker,
                asset_class=future_proto.asset_type, # Might need mapping to Nautilus AssetClass
                currency=currency,
                price_precision=price_precision,
                size_precision=size_precision,
                multiplier=contract_size, # Basic asset size per contract
                lot_size=future_proto.lot, # Number of contracts in a lot
                underlying=future_proto.basic_asset, # The underlying asset
                activation_ns=activation_ts,
                expiration_ns=expiration_ts,
                # margin_init, margin_maint etc. would come from risk rules
                ts_event=0, # Should be last update timestamp
                ts_init=0, # Should be creation timestamp
            )
        except Exception as e:
            self._log.error(f"Error parsing Tinkoff Future {future_proto.figi} into Nautilus FuturesContract: {e}")
            return None

    def get_tinkoff_instrument(self, figi: str) -> Optional[Any]:
        """
        Retrieve the raw Tinkoff instrument proto data by FIGI.

        Parameters
        ----------
        figi : str
            The FIGI of the instrument.

        Returns
        -------
        Any or None
            The Tinkoff instrument proto data, or None if not found.
        """
        return self._instrument_data_cache.get(figi)

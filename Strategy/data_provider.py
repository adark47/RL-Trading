# data_provider.py

import pandas as pd

from nautilus_trader import TEST_DATA_DIR
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider


def prepare_data_1min():
    # Define exchange name
    VENUE_NAME = "XCME"

    # Instrument definition for EURUSD futures (March 2024)
    _INSTRUMENT = TestInstrumentProvider.eurusd_future(
        expiry_year=2024,
        expiry_month=3,
        venue_name=VENUE_NAME,
    )

    # CSV file containing 1-minute bars instrument data above
    csv_file_path = rf"data.csv"

    # Load raw data from CSV file and restructure them into required format for BarDataWrangler
    df = pd.read_csv(csv_file_path, header=0, index_col=False)
    df = df.reindex(columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={"date": "timestamp"})
    df = df.set_index("timestamp")

    # Define bar type
    _1MIN_BARTYPE = BarType.from_str(f"{_INSTRUMENT.id}-1-MINUTE-LAST-EXTERNAL")

    # Convert DataFrame rows into Bar objects
    wrangler = BarDataWrangler(_1MIN_BARTYPE, _INSTRUMENT)
    bars_list: list[Bar] = wrangler.process(df)

    # Collect and return all prepared data
    prepared_data = {
        "venue_name": VENUE_NAME,
        "instrument": _INSTRUMENT,
        "bar_type": _1MIN_BARTYPE,
        "bars_list": bars_list,
    }
    return prepared_data

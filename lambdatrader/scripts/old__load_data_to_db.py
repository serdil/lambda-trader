import csv
import glob
import os

from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.models.candlestick import Candlestick
from lambdatrader.utilities.utils import pair_from, get_project_directory

files = glob.glob(os.path.join(os.path.abspath(get_project_directory()), 'data/') + '*.csv')

store = ChunkCachingCandlestickStore.get_for_exchange(POLONIEX)

for file_path in files:
    currency_name = file_path[file_path.index('BTC_') + 4: file_path.index('.csv')]

    pair = pair_from('BTC', currency_name)

    with open(file_path) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            date = int(row['date'])
            high = float(row['high'])
            low = float(row['low'])
            _open = float(row['open'])
            close = float(row['close'])
            volume = float(row['volume'])
            quote_volume = float(row['quoteVolume'])
            weighted_average = float(row['weightedAverage'])

            store.append_candlestick(
                pair,
                Candlestick(
                    _open=_open,
                    close=close,
                    high=high,
                    low=low,
                    base_volume=volume,
                    quote_volume=quote_volume, date=date,
                    weighted_average=weighted_average
                )
            )

store.persist_chunks()  # TODO Do this in CandlestickStore.

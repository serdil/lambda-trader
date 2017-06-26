import csv
import glob
import os

from backtesting.marketinfo import BacktestMarketInfo
from history.store import CandlestickStore

from models.candlestick import Candlestick
from utils import pair_from, get_project_directory

market_info = BacktestMarketInfo()

files = glob.glob(os.path.join(os.path.abspath(get_project_directory()), 'data/') + '*.csv')

store = CandlestickStore()

for filepath in files:
    currency_name = filepath[filepath.index('BTC_') + 4: filepath.index('.csv')]

    pair = pair_from('BTC', currency_name)

    with open(filepath) as f:
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

store.persist_chunks()

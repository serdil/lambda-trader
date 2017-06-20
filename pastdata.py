import csv
import glob

from marketinfo import BacktestMarketInfo
from models.candlestick import Candlestick
from models.pairinfo import PairInfo
from utils import pair_from


def get_past_market_info() -> BacktestMarketInfo:
    market_info = BacktestMarketInfo()

    files = glob.glob('./data/*.csv')

    for filepath in files:
        currency_name = filepath[filepath.index('BTC_') + 4: filepath.index('.csv')]

        pair = pair_from('BTC', currency_name)
        pair_info = PairInfo(pair)
        market_info.add_pair(pair_info)

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

                pair_info.add_candlestick(Candlestick(_open, close, high, low, volume, quote_volume, date))

    return market_info

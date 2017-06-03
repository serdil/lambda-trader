import csv
import glob

from candlestick import Candlestick
from currency import Currency
from currencypair import CurrencyPair
from marketinfo import MarketInfo
from pairinfo import PairInfo


def get_currency_enum(currency_name) -> Currency:
    enums = {
        'DASH': Currency.DASH,
        'ETC': Currency.ETC,
        'ETH': Currency.ETH,
        'OMNI': Currency.OMNI,
        'REP': Currency.REP
    }

    return enums[currency_name]


def get_past_market_info() -> MarketInfo:
    market_info = MarketInfo()

    files = glob.glob('./data/*.csv')

    for filepath in files:
        currency_name = filepath[filepath.index('BTC_') + 4: filepath.index('.csv')]
        currency = get_currency_enum(currency_name)

        pair = CurrencyPair(Currency.BTC, currency)
        pair_info = PairInfo(pair)
        market_info.add_pair(pair_info)

        with open(filepath) as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                date = int(row['date'])
                high = float(row['high'])
                low = float(row['high'])
                _open = float(row['high'])
                close = float(row['high'])
                volume = float(row['high'])

                pair_info.add_candlestick(Candlestick(_open, close, high, low, volume, date))

    return market_info
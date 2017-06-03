import csv
import glob

from candlestick import Candlestick
from currency import Currency
from currencypair import CurrencyPair
from marketinfo import MarketInfo
from pairinfo import PairInfo


def get_currency_enum(currency_name) -> Currency:
    enums = {
        'BTC': Currency.BTC,
        'AMP': Currency.AMP,
        'ARDR': Currency.ARDR,
        'BCN': Currency.BCN,
        'BCY': Currency.BCY,
        'BELA': Currency.BELA,
        'BLK': Currency.BLK,
        'BTCD': Currency.BTCD,
        'BTM': Currency.BTM,
        'BTS': Currency.BTS,
        'BURST': Currency.BURST,
        'CLAM': Currency.CLAM,
        'DASH': Currency.DASH,
        'DCR': Currency.DCR,
        'DGB': Currency.DGB,
        'DOGE': Currency.DOGE,
        'EMC2': Currency.EMC2,
        'ETC': Currency.ETC,
        'ETH': Currency.ETH,
        'EXP': Currency.EXP,
        'FCT': Currency.FCT,
        'FLDC': Currency.FLDC,
        'FLO': Currency.FLO,
        'GAME': Currency.GAME,
        'GNO': Currency.GNO,
        'GNT': Currency.GNT,
        'GRC': Currency.GRC,
        'HUC': Currency.HUC,
        'LBC': Currency.LBC,
        'LSK': Currency.LSK,
        'LTC': Currency.LTC,
        'MAID': Currency.MAID,
        'NAUT': Currency.NAUT,
        'NAV': Currency.NAV,
        'NEOS': Currency.NEOS,
        'NMC': Currency.NMC,
        'NOTE': Currency.NOTE,
        'NXC': Currency.NXC,
        'NXT': Currency.NXT,
        'OMNI': Currency.OMNI,
        'PASC': Currency.PASC,
        'PINK': Currency.PINK,
        'POT': Currency.POT,
        'PPC': Currency.PPC,
        'RADS': Currency.RADS,
        'REP': Currency.REP,
        'RIC': Currency.RIC,
        'SBD': Currency.SBD,
        'SC': Currency.SC,
        'SJCX': Currency.SJCX,
        'STEEM': Currency.STEEM,
        'STRAT': Currency.STRAT,
        'STR': Currency.STR,
        'SYS': Currency.SYS,
        'VIA': Currency.VIA,
        'VRC': Currency.VRC,
        'VTC': Currency.VTC,
        'XBC': Currency.XBC,
        'XCP': Currency.XCP,
        'XEM': Currency.XEM,
        'XMR': Currency.XMR,
        'XPM': Currency.XPM,
        'XRP': Currency.XRP,
        'XVC': Currency.XVC,
        'ZEC': Currency.ZEC,
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
                low = float(row['low'])
                _open = float(row['open'])
                close = float(row['close'])
                volume = float(row['volume'])
                quote_volume = float(row['quoteVolume'])

                pair_info.add_candlestick(Candlestick(_open, close, high, low, volume, quote_volume, date))

    return market_info
from lambdatrader.constants import M5
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.exchanges.enums import ExchangeEnum


class BaseMarketInfo:

    def get_exchange(self) -> ExchangeEnum:
        raise NotImplementedError

    def get_market_date(self):
        raise NotImplementedError

    @property
    def market_date(self):
        raise NotImplementedError

    def get_active_pairs(self, return_usdt_btc=False):
        raise NotImplementedError

    def get_pair_ticker(self, pair):
        raise NotImplementedError

    def get_pair_last_24h_btc_volume(self, pair):
        raise NotImplementedError

    def get_pair_last_24h_high(self, pair):
        raise NotImplementedError

    def get_pair_latest_candlestick(self, pair, period=M5):
        raise NotImplementedError

    def get_pair_candlestick(self, pair, ind, period=M5, allow_lookahead=False):
        raise NotImplementedError

    def is_candlesticks_supported(self):
        raise NotImplementedError

    def on_pair_tick(self, handler):
        raise NotImplementedError

    def on_all_pairs_tick(self, handler):
        raise NotImplementedError

    def on_pair_candlestick(self, handler):
        raise NotImplementedError

    def on_all_pairs_candlestick(self, handler):
        raise NotImplementedError

    def fetch_ticker(self):
        raise NotImplementedError

    def get_indicator(self, pair, indicator: IndicatorEnum, args, ind=0, period=M5):
        raise NotImplementedError

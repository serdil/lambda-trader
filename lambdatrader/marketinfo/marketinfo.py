from lambdatrader.models.enums.exchange import ExchangeEnum


class BaseMarketInfo:

    def get_exchange(self) -> ExchangeEnum:
        raise NotImplementedError

    def get_market_date(self):
        raise NotImplementedError

    def get_active_pairs(self):
        raise NotImplementedError

    def get_pair_ticker(self, pair):
        raise NotImplementedError

    def get_pair_last_24h_btc_volume(self, pair):
        raise NotImplementedError

    def get_pair_last_24h_high(self, pair):
        raise NotImplementedError

    def get_pair_latest_candlestick(self, pair):
        raise NotImplementedError

    def get_pair_candlestick(self, pair, ind):
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

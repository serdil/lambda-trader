
class PolxMarketInfo:

    def __init__(self):
        pass

    def get_market_time(self):
        pass

    def set_market_time(self, timestamp):
        raise NotImplementedError()

    def get_pair_candlestick(self, currency_pair, ind=0):
        raise NotImplementedError()

    def get_pair_latest_candlestick(self, currency_pair):
        raise NotImplementedError()

    def get_pair_ticker(self, currency_pair):
        pass

    def get_pair_last_24h_btc_volume(self, currency_pair):
        pass

    def get_pair_last_24h_high(self, currency_pair):
        pass

    def pairs(self):
        pass

    def fetcher(self):
        pass


class PolxAccount:

    def __init__(self):
        pass

    def sell(self, currency, amount, market_info):
        pass

    def buy(self, currency, amount, market_info):
        pass

    def new_order(self, order):
        pass

    def new_sequential_fill_order_transaction(self, orders, timeout):
        pass

    def get_order(self, order):
        pass

    def cancel_order(self, order):
        pass

    def get_open_orders(self):
        pass

    def get_open_and_pending_orders(self):
        pass

    def get_estimated_balance(self):
        pass

    def sample_balance(self):
        pass

    def max_avg_drawback(self):
        pass

    def fetcher(self):
        pass

class MarketInfo:
    def __init__(self, pair_infos=[]):
        self.market_time = 0
        self.__pairs = {}
        for pair_info in pair_infos:
            self.add_pair(pair_info)

    def add_pair(self, pair_info):
        self.__pairs[pair_info.currency_pair] = pair_info

    def set_market_time(self, timestamp):
        self.market_time = timestamp

    def get_market_time(self):
        return self.market_time

    def inc_market_time(self):
        self.market_time += 300

    def get_pair_candlestick(self, currency_pair, ind=0):
        return self.__pairs[currency_pair].get_candlestick(self.market_time - ind * 300)

    def get_pair_latest_candlestick(self, currency_pair):
        return self.get_pair_candlestick(currency_pair, 0)

    def get_min_pair_start_time(self):
        return min(map(lambda v: v.get_start_time(), self.__pairs.values()))

    def get_max_pair_end_time(self):
        return max(map(lambda v: v.get_end_time(), self.__pairs.values()))

    def pairs(self):
        return filter(lambda p: p[1].get_start_time() < self.get_market_time() < p[1].get_end_time(), self.__pairs.items())

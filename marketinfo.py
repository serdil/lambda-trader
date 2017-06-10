
class MarketInfo:
    def __init__(self, pair_infos=[]):
        self.market_time = 0
        self.__pairs = {}
        self.__last_volume_calc_timestamp = {}
        self.__last_volume_calc_volume = {}
        self.__last_high_calc_timestamp = {}
        self.__last_high_calc_high = {}
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

    def get_pair_ticker(self, currency_pair):
        pass

    def get_pair_last_24h_btc_volume(self, currency_pair):
        if currency_pair in self.__last_volume_calc_timestamp:
            if self.__last_volume_calc_timestamp[currency_pair] == self.get_market_time():
                return self.__last_volume_calc_volume[currency_pair]
            elif self.__last_volume_calc_timestamp[currency_pair] == self.get_market_time() - 300:
                total_volume = self.__last_volume_calc_volume[currency_pair]
                try:
                    total_volume -= self.get_pair_candlestick(currency_pair, 24 * 12).volume
                except KeyError:
                    pass
                total_volume += self.get_pair_latest_candlestick(currency_pair).volume
                self.__last_volume_calc_timestamp[currency_pair] = self.get_market_time()
                self.__last_volume_calc_volume[currency_pair] = total_volume
                return total_volume

        total_volume = 0.0
        for i in range(24 * 12):
            try:
                total_volume += self.get_pair_candlestick(currency_pair, i).volume
            except KeyError:
                pass
        self.__last_volume_calc_timestamp[currency_pair] = self.get_market_time()
        self.__last_volume_calc_volume[currency_pair] = total_volume
        return total_volume

    def get_pair_last_24h_high(self, currency_pair):
        if currency_pair in self.__last_high_calc_timestamp:
            if self.__last_high_calc_timestamp[currency_pair] == self.get_market_time():
                return self.__last_high_calc_high[currency_pair]
            elif self.__last_high_calc_timestamp[currency_pair] == self.get_market_time() - 300:
                high = self.__last_high_calc_high[currency_pair]
                try:
                    high_omitted = self.get_pair_candlestick(currency_pair, 24 * 12).high
                    if high_omitted > high:
                        del self.__last_high_calc_timestamp[currency_pair]
                        del self.__last_high_calc_high[currency_pair]
                        return self.get_pair_last_24h_high(currency_pair)
                except KeyError:
                    pass
                added_high = self.get_pair_latest_candlestick(currency_pair).high
                high = max(high, added_high)
                self.__last_high_calc_timestamp[currency_pair] = self.get_market_time()
                self.__last_high_calc_high[currency_pair] = high
                return high

        high = 0.0
        for i in range(24 * 12):
            try:
                high = max(high, self.get_pair_candlestick(currency_pair, i).high)
            except KeyError:
                pass
        self.__last_high_calc_timestamp[currency_pair] = self.get_market_time()
        self.__last_high_calc_high[currency_pair] = high
        return high

    def get_min_pair_start_time(self):
        return min(map(lambda v: v.get_start_time(), self.__pairs.values()))

    def get_max_pair_start_time(self):
        return max(map(lambda v: v.get_start_time(), self.__pairs.values()))

    def get_max_pair_end_time(self):
        return max(map(lambda v: v.get_end_time(), self.__pairs.values()))

    def get_min_pair_end_time(self):
        return min(map(lambda v: v.get_end_time(), self.__pairs.values()))

    def pairs(self):
        return filter(lambda p: p[1].get_start_time() < self.get_market_time() < p[1].get_end_time(), self.__pairs.items())

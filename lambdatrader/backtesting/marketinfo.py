from models.ticker import Ticker


class BacktestMarketInfo:
    def __init__(self, candlestick_store):
        self.market_time = 0
        self.candlestick_store = candlestick_store
        self.__last_volume_calc_timestamp = {}
        self.__last_volume_calc_volume = {}
        self.__last_high_calc_timestamp = {}
        self.__last_high_calc_high = {}

    def set_market_time(self, timestamp):
        self.market_time = timestamp

    def get_market_time(self):
        return self.market_time

    def inc_market_time(self):
        self.market_time += 300

    def get_pair_candlestick(self, pair, ind=0):
        return self.candlestick_store.get_candlestick(pair, self.market_time - ind * 300)

    def get_pair_latest_candlestick(self, currency_pair):
        return self.get_pair_candlestick(currency_pair, 0)

    #  return fake ticker
    def get_pair_ticker(self, currency_pair):
        latest_candlestick = self.get_pair_latest_candlestick(currency_pair)
        close_price = latest_candlestick.close
        last = close_price
        lowest_ask = close_price + close_price * 0.005
        highest_bid = close_price - close_price * 0.005
        base_volume = latest_candlestick.base_volume
        quote_volume = latest_candlestick.quote_volume
        percent_change = None
        high24h = self.get_pair_last_24h_high(currency_pair)
        low24h = None
        return Ticker(lowest_ask=lowest_ask, highest_bid=highest_bid, base_volume=base_volume,
                      quote_volume=quote_volume, percent_change=percent_change, high24h=high24h,
                      low24h=low24h, last=last)

    def get_pair_last_24h_btc_volume(self, currency_pair):
        if currency_pair in self.__last_volume_calc_timestamp:
            if self.__last_volume_calc_timestamp[currency_pair] == self.get_market_time():
                return self.__last_volume_calc_volume[currency_pair]
            elif self.__last_volume_calc_timestamp[currency_pair] == self.get_market_time() - 300:
                total_volume = self.__last_volume_calc_volume[currency_pair]
                try:
                    total_volume -= self.get_pair_candlestick(currency_pair, 24 * 12).base_volume
                except KeyError:
                    pass
                total_volume += self.get_pair_latest_candlestick(currency_pair).base_volume
                self.__last_volume_calc_timestamp[currency_pair] = self.get_market_time()
                self.__last_volume_calc_volume[currency_pair] = total_volume
                return total_volume

        total_volume = 0.0
        for i in range(24 * 12):
            try:
                total_volume += self.get_pair_candlestick(currency_pair, i).base_volume
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
        return min(map(
            lambda p: self.__get_pair_start_time_from_store(p), self.__get_pairs_from_store()))

    def get_max_pair_start_time(self):
        return max(map(
            lambda p: self.__get_pair_start_time_from_store(p), self.__get_pairs_from_store()))

    def get_max_pair_end_time(self):
        return max(map(
            lambda p: self.__get_pair_end_time_from_store(p), self.__get_pairs_from_store()))

    def get_min_pair_end_time(self):
        return min(map(
            lambda p: self.__get_pair_end_time_from_store(p), self.__get_pairs_from_store()))

    def __get_pairs_from_store(self):
        return self.candlestick_store.get_pairs()

    def __get_pair_start_time_from_store(self, pair):
        return self.candlestick_store.get_pair_oldest_date(pair)

    def __get_pair_end_time_from_store(self, pair):
        return self.candlestick_store.get_pair_newest_date(pair)

    def pairs(self):
        return list(
            filter(
                self.__pair_exists_in_current_market_time,
                self.__get_pairs_from_store()
            )
        )

    def __pair_exists_in_current_market_time(self, pair):
        return self.__get_pair_start_time_from_store(pair) < \
               self.get_market_time() < \
               self.__get_pair_end_time_from_store(pair)

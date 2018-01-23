from lambdatrader.constants import M5_SECONDS, M5
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.indicators import Indicators
from lambdatrader.marketinfo import BaseMarketInfo
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.models.ticker import Ticker
from lambdatrader.utilities.utils import date_floor


class BacktestingMarketInfo(BaseMarketInfo):

    def __init__(self, candlestick_store):
        self.__market_date = 0
        self.candlestick_store = candlestick_store
        self.indicators = Indicators(self)

        self.__last_volume_calc_date = {}
        self.__last_volume_calc_volume = {}
        self.__last_high_calc_date = {}
        self.__last_high_calc_high = {}

    def get_exchange(self) -> ExchangeEnum:
        return ExchangeEnum.BACKTESTING

    def set_market_date(self, timestamp):
        self.__market_date = timestamp

    def get_market_date(self):
        return self.market_date

    @property
    def market_date(self):
        return self.__market_date

    def inc_market_date(self):
        self.__market_date += M5_SECONDS

    def get_pair_candlestick(self, pair, ind=0, period=M5):
        if ind < 0:
            raise Exception('Cannot look ahead.')
        try:
            return self.get_pair_period_candlestick(pair, ind=ind, period=period)
        except KeyError as e:
            print('Error while getting candlestick for pair:', pair)
            raise e

    def get_pair_period_candlestick(self, pair, ind, period=M5):
        end_date = date_floor(self.market_date, period=period) - ind * period.seconds()
        start_date = (date_floor(self.market_date, period=period)
                      - (ind+1) * period.seconds() + M5_SECONDS)
        candlestick = self.candlestick_store.get_candlestick(pair=pair, date=start_date)
        for date in range(start_date+M5_SECONDS, end_date+M5_SECONDS, M5_SECONDS):
            next_candlestick = self.candlestick_store.get_candlestick(pair=pair,
                                                                      date=end_date)
            candlestick = candlestick.batch_with(next_candlestick)
        candlestick.period = period
        return candlestick

    def get_pair_latest_candlestick(self, pair, period=M5):
        return self.get_pair_candlestick(pair=pair, ind=0, period=period)

    #  return fake ticker
    def get_pair_ticker(self, pair):
        latest_candlestick = self.get_pair_latest_candlestick(pair=pair)
        close_price = latest_candlestick.close
        last = close_price
        lowest_ask = close_price + close_price * 0.005
        highest_bid = close_price - close_price * 0.005
        base_volume = latest_candlestick.base_volume
        quote_volume = latest_candlestick.quote_volume
        percent_change = None
        high24h = self.get_pair_last_24h_high(pair=pair)
        low24h = None
        return Ticker(lowest_ask=lowest_ask, highest_bid=highest_bid, base_volume=base_volume,
                      quote_volume=quote_volume, percent_change=percent_change, high24h=high24h,
                      low24h=low24h, last=last)

    def get_pair_last_24h_btc_volume(self, pair):
        if pair in self.__last_volume_calc_date:
            if self.__last_volume_calc_date[pair] == self.market_date:
                return self.__last_volume_calc_volume[pair]
            elif self.__last_volume_calc_date[pair] == self.market_date - M5_SECONDS:
                total_volume = self.__last_volume_calc_volume[pair]
                try:
                    total_volume -= self.get_pair_candlestick(pair=pair, ind=24 * 12).base_volume
                except KeyError:
                    pass
                total_volume += self.get_pair_latest_candlestick(pair=pair).base_volume
                self.__last_volume_calc_date[pair] = self.market_date
                self.__last_volume_calc_volume[pair] = total_volume
                return total_volume

        total_volume = 0.0
        for i in range(24 * 12):
            try:
                total_volume += self.get_pair_candlestick(pair=pair, ind=i).base_volume
            except KeyError:
                pass
        self.__last_volume_calc_date[pair] = self.market_date
        self.__last_volume_calc_volume[pair] = total_volume
        return total_volume

    def get_pair_last_24h_high(self, pair):
        return self.get_pair_last_24h_high_cached(pair=pair)

    def get_pair_last_24h_high_cached(self, pair):
        if pair in self.__last_high_calc_date:
            if self.__last_high_calc_date[pair] == self.market_date:
                return self.__last_high_calc_high[pair]
            elif self.__last_high_calc_date[pair] == self.market_date - M5_SECONDS:
                high = self.__last_high_calc_high[pair]
                try:
                    high_omitted = self.get_pair_candlestick(pair=pair, ind=24 * 12).high
                    if high_omitted == high:
                        del self.__last_high_calc_date[pair]
                        del self.__last_high_calc_high[pair]
                        return self.get_pair_last_24h_high_cached(pair=pair)
                except KeyError:
                    pass
                added_high = self.get_pair_latest_candlestick(pair=pair).high
                high = max(high, added_high)
                self.__last_high_calc_date[pair] = self.market_date
                self.__last_high_calc_high[pair] = high
                return high

        high = self.get_pair_last_24h_high_uncached(pair=pair)
        self.__last_high_calc_date[pair] = self.market_date
        self.__last_high_calc_high[pair] = high
        return high

    def get_pair_last_24h_high_uncached(self, pair):
        high = 0.0
        for i in range(24 * 12):
            try:
                high = max(high, self.get_pair_candlestick(pair=pair, ind=i).high)
            except KeyError:
                pass
        return high

    def get_min_pair_start_time(self):
        return min(map(
            lambda p: self.__get_pair_start_time_from_store(pair=p), self.__get_pairs_from_store()))

    def get_max_pair_start_time(self):
        return max(map(
            lambda p: self.__get_pair_start_time_from_store(pair=p), self.__get_pairs_from_store()))

    def get_max_pair_end_time(self):
        return max(map(
            lambda p: self.__get_pair_end_time_from_store(pair=p), self.__get_pairs_from_store()))

    def get_min_pair_end_time(self):
        return min(map(
            lambda p: self.__get_pair_end_time_from_store(pair=p), self.__get_pairs_from_store()))

    def __get_pairs_from_store(self):
        return self.candlestick_store.get_pairs()

    def __get_pair_start_time_from_store(self, pair):
        return self.candlestick_store.get_pair_period_oldest_date(pair=pair)

    def __get_pair_end_time_from_store(self, pair):
        return self.candlestick_store.get_pair_period_newest_date(pair=pair)

    def get_active_pairs(self, return_usdt_btc=False):
        pairs_set = set(
            filter(
                self.__pair_exists_in_current_market_time,
                self.__get_pairs_from_store()
            )
        )
        if not return_usdt_btc and 'USDT_BTC' in pairs_set:
            pairs_set.remove('USDT_BTC')
        return list(pairs_set)

    def is_candlesticks_supported(self):
        return True

    def __pair_exists_in_current_market_time(self, pair):
        return self.__get_pair_start_time_from_store(pair=pair) < \
               self.market_date < \
               self.__get_pair_end_time_from_store(pair=pair)

    def on_pair_candlestick(self, handler):
        raise NotImplementedError

    def on_pair_tick(self, handler):
        raise NotImplementedError

    def on_all_pairs_candlestick(self, handler):
        raise NotImplementedError

    def on_all_pairs_tick(self, handler):
        raise NotImplementedError

    def fetch_ticker(self):
        pass

    def get_indicator(self, pair, indicator: IndicatorEnum, args, ind=0, period=M5):
        return self.indicators.compute(pair, indicator, args, ind=ind, period=period)

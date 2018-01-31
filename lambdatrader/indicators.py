import numpy
import numpy as np
from lambdatrader.indicator_functions import IndicatorEnum

from lambdatrader.constants import M5
from lambdatrader.marketinfo import BaseMarketInfo
from lambdatrader.signals.utils import (
    candlesticks_open, candlesticks_high, candlesticks_low, candlesticks_close, candlesticks_volume,
)


class Indicators:

    def __init__(self, market_info: BaseMarketInfo):
        self._indicator_cache = {}
        self.market_info = market_info

    def compute(self, pair, indicator: IndicatorEnum, args, ind=0, period=M5):
        cache_key = self.cache_key(self.ind_date(ind, period=period), indicator, args, period)
        if cache_key not in self._indicator_cache:
            self.precompute_and_cache(pair, indicator, args, ind, period)
        return self._indicator_cache[cache_key]

    def precompute_and_cache(self, pair, indicator: IndicatorEnum, args, ind, period):
        start_ind, end_ind, indicator_input = self.get_input_for_range(pair, ind, period)
        results = self.get_indicator_results(indicator_input, indicator, args)

        for i, date in enumerate(self.iter_ind_dates(start_ind, end_ind, period)):
            date_results = []
            for range_result in results:
                date_result = range_result[i]
                if numpy.isnan(date_result):
                    date_results = None
                    break
                date_results.append(date_result)
            if date_results:
                cache_key = self.cache_key(date, indicator, args, period)
                self._indicator_cache[cache_key] = tuple(date_results)

    @staticmethod
    def get_indicator_results(indicator_input, indicator, args):
        indicator_function = indicator.function()
        results = indicator_function(indicator_input, *args)
        if not isinstance(results, list):
            results = results,
        return tuple(results)

    def get_input_for_range(self, pair, ind, period, pre_offset=250, post_offset=1000):
        now_and_backwards_candles = []
        try:
            for i in range(ind, ind+pre_offset+1):
                candle = self.market_info.get_pair_candlestick(pair, ind=i,
                                                               period=period, allow_lookahead=True)
                now_and_backwards_candles.append(candle)
        except KeyError:
            print('Indicators: Error while getting candlestick for pair:', pair)
        start_ind = ind + len(now_and_backwards_candles) - 1
        forward_candles = []
        try:
            for i in range(ind-1, ind-post_offset-1, -1):
                candle = self.market_info.get_pair_candlestick(pair, ind=i,
                                                               period=period, allow_lookahead=True)
                forward_candles.append(candle)
        except KeyError:
            print('Indicators: Error while getting candlestick for pair:', pair)
        end_ind = ind - len(forward_candles) - 1
        all_candles = now_and_backwards_candles[::-1] + forward_candles
        input_dict = {
            'open': np.array(candlesticks_open(all_candles)),
            'high': np.array(candlesticks_high(all_candles)),
            'low': np.array(candlesticks_low(all_candles)),
            'close': np.array(candlesticks_close(all_candles)),
            'volume': np.array(candlesticks_volume(all_candles))
        }
        return start_ind, end_ind, input_dict

    @property
    def market_date(self):
        return self.market_info.market_date

    def iter_ind_dates(self, start_ind, end_int, period):
        for ind in range(start_ind, end_int, -1):
            yield self.ind_date(ind, period)

    def ind_date(self, ind, period):
        return self.market_date - ind*period.seconds()

    @staticmethod
    def cache_key(market_date, indicator, args, period):
        return market_date, indicator.name, tuple(args), period.name

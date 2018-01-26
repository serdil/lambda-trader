import numpy as np
from lambdatrader.indicator_functions import IndicatorEnum

from lambdatrader.constants import M5
from lambdatrader.marketinfo import BaseMarketInfo
from lambdatrader.signals.utils import (
    candlesticks_open, candlesticks_high, candlesticks_low, candlesticks_close, candlesticks_volume,
)


class Indicators:

    def __init__(self, market_info: BaseMarketInfo):
        self.market_info = market_info

    def compute(self, pair, indicator: IndicatorEnum, args, ind=0, period=M5):
        indicator_function = indicator.function()
        indicator_input = self.get_input(pair=pair, ind=ind, period=period)
        range_results = indicator_function(indicator_input, *args)
        results_list = []
        for range_result in range_results:
            results_list.append(range_result[-1])
        return tuple(results_list)

    def get_input(self, pair, ind=0, period=M5, num_candles=100):
        input_candles = []
        for i in range(ind+num_candles-1, ind-1, -1):
            candle = self.market_info.get_pair_candlestick(pair, ind=i, period=period)
            input_candles.append(candle)
        return {
            'open': np.array(candlesticks_open(input_candles)),
            'high': np.array(candlesticks_high(input_candles)),
            'low': np.array(candlesticks_low(input_candles)),
            'close': np.array(candlesticks_close(input_candles)),
            'volume': np.array(candlesticks_volume(input_candles))
        }

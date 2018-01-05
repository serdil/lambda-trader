import numpy as np

from lambdatrader.constants import M5, IndicatorEnum
from lambdatrader.marketinfo import BaseMarketInfo


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
            'open': np.array(self.candlesticks_open(input_candles)),
            'high': np.array(self.candlesticks_high(input_candles)),
            'low': np.array(self.candlesticks_low(input_candles)),
            'close': np.array(self.candlesticks_close(input_candles)),
            'volume': np.array(self.candlesticks_volume(input_candles))
        }

    @classmethod
    def candlesticks_open(cls, candlesticks):
        return cls.list_map(cls.candlestick_open, candlesticks)

    @classmethod
    def candlesticks_high(cls, candlesticks):
        return cls.list_map(cls.candlestick_high, candlesticks)

    @classmethod
    def candlesticks_low(cls, candlesticks):
        return cls.list_map(cls.candlestick_low, candlesticks)

    @classmethod
    def candlesticks_close(cls, candlesticks):
        return cls.list_map(cls.candlestick_close, candlesticks)

    @classmethod
    def candlesticks_volume(cls, candlesticks):
        return cls.list_map(cls.candlestick_volume, candlesticks)

    @staticmethod
    def list_map(func, items):
        return list(map(func, items))

    @staticmethod
    def candlestick_open(candlestick):
        return candlestick.open

    @staticmethod
    def candlestick_high(candlestick):
        return candlestick.high

    @staticmethod
    def candlestick_low(candlestick):
        return candlestick.low

    @staticmethod
    def candlestick_close(candlestick):
        return candlestick.close

    @staticmethod
    def candlestick_volume(candlestick):
        return candlestick.base_volume

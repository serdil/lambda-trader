from typing import List

from lambdatrader.indicator_functions import IndicatorEnum


def candlesticks_open(candlesticks):
    return list_map(candlestick_open, candlesticks)


def candlesticks_high(candlesticks):
    return list_map(candlestick_high, candlesticks)


def candlesticks_low(candlesticks):
    return list_map(candlestick_low, candlesticks)


def candlesticks_close(candlesticks):
    return list_map(candlestick_close, candlesticks)


def candlesticks_volume(candlesticks):
    return list_map(candlestick_volume, candlesticks)


def list_map(func, items):
    return list(map(func, items))


def candlestick_open(candlestick):
    return candlestick.open


def candlestick_high(candlestick):
    return candlestick.high


def candlestick_low(candlestick):
    return candlestick.low


def candlestick_close(candlestick):
    return candlestick.close


def candlestick_volume(candlestick):
    return candlestick.base_volume


def get_candle(market_info, pair, ind, period):
    return market_info.get_pair_period_candlestick(pair, ind, period=period)


def get_indicator(market_info, pair, indicator: IndicatorEnum, args: List, ind, period):
    ind_output = market_info.get_indicator(pair, indicator, args, ind, period=period)
    try:
        return list(ind_output)
    except TypeError:
        return [ind_output]

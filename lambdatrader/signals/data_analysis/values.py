from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.constants import M5, PeriodEnum


def make_close_price_in_fifteen_mins():
    return make_close_price_in_future(num_candles=3, candle_period=M5)


def make_max_price_in_fifteen_mins():
    return make_max_price_in_future(num_candles=3, candle_period=M5)


def make_close_price_in_future(num_candles, candle_period: PeriodEnum):
    def value_close_price_in_future(market_info: BacktestingMarketInfo, pair):
        last_candle = market_info.get_pair_period_candlestick(pair, 0, period=candle_period)
        candle_in_future = market_info.get_pair_period_candlestick(pair=pair,
                                                                   ind=-num_candles,
                                                                   period=candle_period)
        return candle_in_future.close / last_candle.close

    return value_close_price_in_future


def make_max_price_in_future(num_candles, candle_period: PeriodEnum):
    def value_max_price_in_future(market_info: BacktestingMarketInfo, pair):
        last_candle = market_info.get_pair_period_candlestick(pair, 0, period=candle_period)
        highest_until_future = float('-inf')
        for i in range(-1, -num_candles-1, -1):
            candle = market_info.get_pair_period_candlestick(pair=pair,
                                                             ind=-i,
                                                             period=candle_period)
            highest_until_future = max(highest_until_future, candle.high)
        return highest_until_future / last_candle.close

    return value_max_price_in_future

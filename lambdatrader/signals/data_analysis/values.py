from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.constants import M5, PeriodEnum

INCREASE_MOON = 0.20
INCREASE_UP = 0.03
INCREASE_FLAT_DOWN = 0.00

CLASS_INCREASE_MOON = 'moon'
CLASS_INCREASE_UP = 'up'
CLASS_INCREASE_FLAT_DOWN = 'flat'


def make_cont_close_price_in_one_hour():
    return make_cont_close_price_in_future(num_candles=12, candle_period=M5)


def make_cont_close_price_in_fifteen_mins():
    return make_cont_close_price_in_future(num_candles=3, candle_period=M5)


def make_class_max_price_in_fifteen_mins():
    return make_class_max_price_in_future(num_candles=3, candle_period=M5)


def make_binary_max_price_in_future(increase=0.01, num_candles=3, candle_period: PeriodEnum=M5):
    def value_binary_max_price_in_future(market_info: BacktestingMarketInfo, pair):
        cont_func = make_cont_max_price_in_future(num_candles=num_candles,
                                                  candle_period=candle_period)
        cont_value = cont_func(market_info, pair)
        if cont_value >= increase:
            return 1
        else:
            return 0
    return value_binary_max_price_in_future


def make_class_max_price_in_future(num_candles, candle_period: PeriodEnum):
    def value_class_max_price_in_future(market_info: BacktestingMarketInfo, pair):
        cont_func = make_cont_max_price_in_future(num_candles=num_candles,
                                                  candle_period=candle_period)
        cont_value = cont_func(market_info, pair)
        if cont_value >= INCREASE_MOON:
            return CLASS_INCREASE_MOON
        elif cont_value >= INCREASE_UP:
            return CLASS_INCREASE_UP
        else:
            return CLASS_INCREASE_FLAT_DOWN
    return value_class_max_price_in_future


def make_cont_max_price_in_fifteen_mins():
    return make_cont_max_price_in_future(num_candles=3, candle_period=M5)


def make_cont_close_price_in_future(num_candles, candle_period: PeriodEnum):
    def value_cont_close_price_in_future(market_info: BacktestingMarketInfo, pair):
        last_candle = market_info.get_pair_period_candlestick(pair, 0, period=candle_period)
        candle_in_future = market_info.get_pair_period_candlestick(pair=pair,
                                                                   ind=-num_candles,
                                                                   period=candle_period)
        return candle_in_future.close / last_candle.close - 1.0

    return value_cont_close_price_in_future


def make_cont_max_price_in_future(num_candles, candle_period: PeriodEnum):
    def value_cont_max_price_in_future(market_info: BacktestingMarketInfo, pair):
        last_candle = market_info.get_pair_period_candlestick(pair, 0, period=candle_period)
        highest_until_future = float('-inf')
        for i in range(-1, -num_candles-1, -1):
            candle = market_info.get_pair_period_candlestick(pair=pair,
                                                             ind=i,
                                                             period=candle_period)
            highest_until_future = max(highest_until_future, candle.high)
        return highest_until_future / last_candle.close - 1.0

    return value_cont_max_price_in_future


def make_cont_min_price_in_future(num_candles, candle_period: PeriodEnum):
    def value_cont_min_price_in_future(market_info: BacktestingMarketInfo, pair):
        last_candle = market_info.get_pair_period_candlestick(pair, 0, period=candle_period)
        lowest_until_future = float('inf')
        for i in range(-1, -num_candles-1, -1):
            candle = market_info.get_pair_period_candlestick(pair=pair,
                                                             ind=i,
                                                             period=candle_period)
            lowest_until_future = min(lowest_until_future, candle.high)
        return lowest_until_future / last_candle.close - 1.0

    return value_cont_min_price_in_future


def make_cont_trade_return(num_candles=3, tp_level=0.03, candle_period=M5):
    def value_cont_trade_return(market_info: BacktestingMarketInfo, pair):
        max_price_func = make_cont_max_price_in_future(num_candles=num_candles,
                                                       candle_period=candle_period)
        future_price_func = make_cont_close_price_in_future(num_candles=num_candles,
                                                            candle_period=candle_period)
        max_price = max_price_func(market_info, pair)
        future_price = future_price_func(market_info, pair)
        tp_hit = max_price >= tp_level
        if tp_hit:
            return tp_level
        else:
            return future_price

    return value_cont_trade_return


def value_dummy(market_info, pair):
    return 0

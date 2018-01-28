from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.constants import M5


def value_price_after_fifteen_min(market_info: BacktestingMarketInfo, pair):
    close_after_fifteen_mins = market_info.get_pair_period_candlestick(pair, -3, period=M5).close
    close_now = market_info.get_pair_period_candlestick(pair, 0, period=M5).close
    return close_after_fifteen_mins / close_now

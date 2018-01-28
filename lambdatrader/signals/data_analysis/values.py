from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.constants import M5


def value_price_after_fifteen_min(market_info: BacktestingMarketInfo, pair):
    return market_info.get_pair_period_candlestick(pair, -3, period=M5)

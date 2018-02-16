from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import ChunkCachingCandlestickStore
from lambdatrader.constants import H4, D
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.utilities.utils import seconds

market_info = BacktestingMarketInfo(candlestick_store=ChunkCachingCandlestickStore.get_for_exchange(POLONIEX))

market_date = market_info.get_max_pair_end_time() - seconds(days=7)

market_info.set_market_date(market_date)

macd_vals = market_info.get_indicator('BTC_ETH', IndicatorEnum.MACD, [12, 26, 9], period=H4)

print(macd_vals)

for i in range(365, -1, -1):
    macd_vals = market_info.get_indicator('BTC_ETH', IndicatorEnum.MACD, [12, 26, 9],
                                          ind=i, period=D)
    print(macd_vals)

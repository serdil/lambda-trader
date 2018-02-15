from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import CandlestickStore
from lambdatrader.constants import M5, M5_SECONDS
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.indicator_functions import IndicatorEnum

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

# latest_safe_date = market_info.get_min_pair_end_time()
#
# market_date = latest_safe_date - seconds(hours=1)
#
# market_info.set_market_date(market_date)
#
# print('request sma 251 for date:', market_date)
# sma_value = market_info.get_indicator('BTC_ETH', IndicatorEnum.SMA, [251], ind=0, period=M5)
# print(sma_value)


earliest_eth_date = market_info.get_pair_start_time('BTC_ETH')

market_info.set_market_date(earliest_eth_date)

print('request BTC_ETH sma 3 for date:', earliest_eth_date + 2*M5_SECONDS)

sma_value = market_info.get_indicator('BTC_ETH', IndicatorEnum.SMA, [3], ind=-2, period=M5)
print(sma_value)

import time

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import CandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.utilities.utils import seconds

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

market_info.set_market_date(market_info.get_min_pair_start_time() + seconds(years=1))

current_date = 0

pairs = market_info.get_active_pairs()
start_time = time.time()
for cycle in range(200):
    for pair in pairs:
        for i in range(100):
            try:
                market_info.get_pair_candlestick(pair=pair, ind=i, period=M5)
            except KeyError:
                pass
    market_info.inc_market_date()
print('elapsed time:', time.time() - start_time)

import time

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.utilities.utils import seconds

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_start_date = store.get_pair_period_oldest_date(symbol)
symbol_end_date = store.get_pair_period_newest_date(symbol)

small_start_date = symbol_end_date - int(seconds(days=1))
range_small = store.get_candlestick_range(symbol,
                                          start_date=small_start_date,
                                          end_date=symbol_end_date)

start_time = time.time()

range_large = store.get_candlestick_range(symbol,
                                          start_date=symbol_start_date,
                                          end_date=symbol_end_date)

duration = time.time() - start_time

print(len(range_large))
print('duration:', duration)

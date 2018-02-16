import time

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.utilities.utils import seconds

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date('BTC_ETH')

small_start_date = symbol_end_date - int(seconds(days=1))
df_small = store.get_df(symbol, start_date=small_start_date)


start_time = time.time()

df_large = store.get_df(symbol)

duration = time.time() - start_time

print(df_large.describe)
print('duration:', duration)

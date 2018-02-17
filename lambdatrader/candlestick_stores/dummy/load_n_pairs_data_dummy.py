import time

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbols = list(store.get_pairs())
n_symbols = 100

total_len = 0

start_time = time.time()

df_list = []

for i, symbol in enumerate(symbols[:n_symbols]):
    print(i, symbol)
    df = store.get_df(symbol)
    df_list.append(df)
    total_len += len(df)


duration = time.time() - start_time

print()
print(total_len)
print('duration:', duration)

time.sleep(10)

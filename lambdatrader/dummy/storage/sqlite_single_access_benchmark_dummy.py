import os
import sqlite3
import time

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import ChunkCachingCandlestickStore
from lambdatrader.config import CANDLESTICK_DB_DIRECTORY
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.utilities.utils import seconds


def get_latest_market_date():
    market_info = BacktestingMarketInfo(
        candlestick_store=ChunkCachingCandlestickStore.get_for_exchange(POLONIEX))

    return market_info.get_max_pair_end_time()


DATABASE_DIR = CANDLESTICK_DB_DIRECTORY

db_path = os.path.join(DATABASE_DIR, '{}.db'.format(POLONIEX.name))

conn = sqlite3.connect(db_path)

cursor = conn.cursor()

symbol = 'BTC_ETH:M5'


latest_market_date = get_latest_market_date()
day_offset = 100
num_days = 500

start_date = int(latest_market_date - seconds(days=day_offset + num_days))
end_date = int(latest_market_date - seconds(days=day_offset))

num_fetch = seconds(days=num_days) // M5.seconds()

start_time = time.time()

for date in range(start_date, end_date, M5.seconds()):
    query = "SELECT * FROM '{}' WHERE date == ?".format(symbol)
    cursor.execute(query, (date,))
    for row in cursor:
        pass

duration = time.time() - start_time

print(num_days, num_fetch)
print(duration)

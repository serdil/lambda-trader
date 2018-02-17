from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.constants import OHLCV_CLOSE
from lambdatrader.signals.data_analysis.df_features import OHLCVCloseDelta
from lambdatrader.utilities.utils import seconds

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

small_start_date = symbol_end_date - int(seconds(days=1))
df_small = store.get_df(symbol, start_date=small_start_date)

close_delta_feature = OHLCVCloseDelta(OHLCV_CLOSE, offset=1, period=M5)

print(df_small)

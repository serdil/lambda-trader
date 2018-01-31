from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.features import get_feature_funcs_iter
from lambdatrader.signals.data_analysis.values import make_cont_max_price_in_fifteen_mins
from lambdatrader.utilities.utils import seconds

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

print(market_info)

latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_ETH'
dataset_start_date = latest_market_date - seconds(days=7, minutes=30)
dataset_end_date = latest_market_date - seconds(days=7)

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_feature_funcs_iter()),
                                           value_function=make_cont_max_price_in_fifteen_mins())

print('created dataset')

feature_names = dataset.get_first_feature_names()

X = dataset.get_numpy_feature_matrix()
y = dataset.get_numpy_value_array()

print(feature_names)
print(X)
print(y)

print('X shape:', X.shape)
print('y shape:', y.shape)

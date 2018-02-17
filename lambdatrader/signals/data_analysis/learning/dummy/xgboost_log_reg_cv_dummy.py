import xgboost as xgb

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.shelve_cache import shelve_cache_save
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set_with_indicators,
)
from lambdatrader.signals.data_analysis.values import (
    make_binary_max_price_in_future,
)
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_RADS'

day_offset = 3

dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*365)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*200)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*120)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*90)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*60)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*30)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24)
# dataset_start_date = latest_market_date - seconds(days=day_offset, minutes=30)

dataset_end_date = latest_market_date - seconds(days=day_offset)

dataset_len = dataset_end_date - dataset_start_date

feature_funcs_name = None

increase = 0.03
num_candles = 48
value_func = make_binary_max_price_in_future(increase=increase, num_candles=num_candles)
value_func_name = 'binary_max_price_{}_{}'.format(increase, num_candles)

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_small_feature_func_set_with_indicators()),
                                           value_function=value_func,
                                           cache_and_get_cached=True,
                                           feature_functions_key=feature_funcs_name,
                                           value_function_key=value_func_name)

print('created/loaded dataset\n')

feature_names = dataset.get_first_feature_names()

X = dataset.get_numpy_feature_matrix()
y = dataset.get_numpy_value_array()


dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)


params = {
    'silent': 0,

    'objective': 'binary:logistic',
    'base_score': 0.5,
    'eval_metric': 'error@0.90',

    'eta': 0.3,
    'gamma': 0,
    'max_depth': 6,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0,
    'tree_method': 'auto',
    'sketch_eps': 0.03,
    'scale_pos_weight': 1,
    'updater': 'grow_colmaker,prune',
    'refresh_leaf': 1,
    'process_type': 'default',
    'grow_policy': 'depthwise',
    'max_leaves': 0,
    'max_bin': 256,
    'predictor': 'cpu_predictor',
}

num_round = 1000

cv_fold = 10

cv_out = xgb.cv(params=params,
                dtrain=dtrain,
                num_boost_round=num_round)

train_mean = cv_out['train-error@0.9-mean']
train_std = cv_out['train-error@0.9-std']
test_mean = cv_out['test-error@0.9-mean']
test_std = cv_out['test-error@0.9-std']

print('          {:<25}{:<25}{:<25}{:<25}'.format('train-error-mean', 'train-error-std', 'test-error-mean', 'test-error-std'))
for i, (v1, v2, v3, v4) in enumerate(zip(train_mean, train_std, test_mean, test_std)):
    print('{:<10}{:<25}{:<25}{:<25}{:<25}'.format(i, v1, v2, v3, v4))

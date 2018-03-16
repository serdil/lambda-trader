import random

from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn, CloseReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets, ValueSets
from lambdatrader.signals.data_analysis.models import XGBSplitDatasetModel
from lambdatrader.signals.generators.dummy.feature_spaces import ohlcv_sampler, all_sampler

from lambdatrader.signals.generators.factories import Pairs

# random.seed(0)
# random.seed(1)
# random.seed(2)
# random.seed(3)
# random.seed(4)
# random.seed(5)
# random.seed(6)
# random.seed(7)

# xgb_training_pairs = Pairs.all_pairs(); interleaved = True
# xgb_training_pairs = Pairs.all_pairs()[:40]; interleaved = True
# xgb_training_pairs = Pairs.all_pairs()[:20]; interleaved = True
# xgb_training_pairs = Pairs.all_pairs()[:10]; interleaved = True
# xgb_training_pairs = Pairs.all_pairs()[25:30]; interleaved = True
# xgb_training_pairs = Pairs.all_pairs()[20:25]; interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 40); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 20); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 15); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 10); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 5); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 3); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 1); interleaved = True
# xgb_training_pairs = Pairs.n_pairs(); interleaved = True
# xgb_training_pairs = ['BTC_ETH']; interleaved = False
# xgb_training_pairs = ['BTC_XMR']; interleaved = False
# xgb_training_pairs = ['BTC_LTC']; interleaved = False
xgb_training_pairs = ['BTC_XRP']; interleaved = False
# xgb_training_pairs = ['BTC_STR']; interleaved = False
# xgb_training_pairs = ['BTC_RADS']; interleaved = False
# xgb_training_pairs = ['BTC_RIC']; interleaved = False
# xgb_training_pairs = ['BTC_SC']; interleaved = False
# xgb_training_pairs = ['BTC_VIA']; interleaved = False
# xgb_training_pairs = ['BTC_VTC']; interleaved = False
# xgb_training_pairs = ['BTC_XCP']; interleaved = False
# xgb_training_pairs = ['BTC_XVC']; interleaved = False
# xgb_training_pairs = ['BTC_STEEM']; interleaved = False
# xgb_training_pairs = ['BTC_ZRX']; interleaved = False
# xgb_training_pairs = ['BTC_NAUT']; interleaved = False


# xgb_training_pairs = random.sample(Pairs.all_pairs(), 5); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 3); interleaved = True
# xgb_training_pairs = random.sample(Pairs.all_pairs(), 1); interleaved = True
# xgb_training_pairs.append('BTC_XRP')

# low success rate
# xgb_training_pairs = ['BTC_BTM', 'BTC_VRC', 'BTC_HUC', 'BTC_NXC', 'BTC_NEOS', 'BTC_RIC']; interleaved = True
# xgb_training_pairs = ['BTC_BTM']; interleaved = False
# xgb_training_pairs = ['BTC_VRC']; interleaved = False
# xgb_training_pairs = ['BTC_HUC']; interleaved = False
# xgb_training_pairs = ['BTC_NXC']; interleaved = False
# xgb_training_pairs = ['BTC_NEOS']; interleaved = False
# xgb_training_pairs = ['BTC_RIC']; interleaved = False

# few signals
# xgb_training_pairs = ['BTC_ETC', 'BTC_PINK', 'BTC_FLO', 'BTC_BCH', 'BTC_GNO', 'BTC_FLDC']; interleaved = True
# xgb_training_pairs = ['BTC_ETC']; interleaved = False
# xgb_training_pairs = ['BTC_PINK']; interleaved = False
# xgb_training_pairs = ['BTC_FLO']; interleaved = False
# xgb_training_pairs = ['BTC_BCH']; interleaved = False
# xgb_training_pairs = ['BTC_GNO']; interleaved = False
# xgb_training_pairs = ['BTC_FLDC']; interleaved = False
# xgb_training_pairs = ['BTC_NOTE']; interleaved = False

# good
# xgb_training_pairs = ['BTC_ZRX']; interleaved = False
# xgb_training_pairs = ['BTC_BURST']; interleaved = False

print('training pairs:', xgb_training_pairs)


# xgb_split_date_range = SplitDateRanges.january_3_days_test_3_days_val_7_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_20_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_160_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_360_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_500_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_1000_days_train()
xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_rest_train()

# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=7, t=7)

# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=500)
# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=5000)

# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=20)
# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=200)
# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=500)

# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=60, t=20)
# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=60, t=60)
# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=60, t=200)

# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=200, t=200)
# xgb_split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=200, t=500)

# feature_sampler = ohlcv_sampler
feature_sampler = all_sampler

# feature_set = FeatureSets.get_all_periods_last_ten_ohlcv_now_delta()

# feature_set = feature_sampler.sample(size=5)
feature_set = feature_sampler.sample(size=10)
# feature_set = feature_sampler.sample(size=20)
# feature_set = feature_sampler.sample(size=100)
# feature_set = feature_sampler.sample(size=500)
# feature_set = feature_sampler.sample(size=1000)

xgb_n_candles = 48
value_set_close = DFFeatureSet(features=[CloseAvgReturn(n_candles=xgb_n_candles)])
# value_set_close = DFFeatureSet(features=[CloseReturn(n_candles=xgb_n_candles)])

value_set_max = DFFeatureSet(features=[MaxReturn(n_candles=xgb_n_candles)])

xgb_model_per_pair = False

xgb_c_thr = 0.01
xgb_m_thr = 0.02

close_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=xgb_training_pairs,
    feature_set=feature_set,
    value_set=value_set_close,
    split_date_range=xgb_split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=xgb_training_pairs,
    feature_set=feature_set,
    value_set=value_set_max,
    split_date_range=xgb_split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

pair_close_datasets = {}
pair_max_datasets = {}

for pair in xgb_training_pairs:
    pair_close_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=[pair],
        feature_set=feature_set,
        value_set=value_set_close,
        split_date_range=xgb_split_date_range,
        exchanges=(POLONIEX,),
        interleaved=False
    )
    pair_max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=[pair],
        feature_set=feature_set,
        value_set=value_set_max,
        split_date_range=xgb_split_date_range,
        exchanges=(POLONIEX,),
        interleaved=False
    )
    pair_close_datasets[pair] = pair_close_dataset
    pair_max_datasets[pair] = pair_max_dataset

xgb_params = {
    'silent': 1,
    'booster': 'gbtree',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.03,
    'gamma':0,
    'max_depth': 4,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'tree_method': 'exact',
    'sketch_eps': 0.03,
    'scale_pos_weight': 1,
    'refresh_leaf': 1,
    'process_type': 'default',
    'grow_policy': 'depthwise',
    'max_leaves': 0,
    'max_bin': 256,

    'lambda': 1,
    'alpha': 0,
    'updater': 'grow_colmaker,prune',

    # 'sample_type': 'uniform',
    # 'normalize_type': 'tree',
    # 'rate_drop': 0.00,
    # 'one_drop': 0,
    # 'skip_drop': 0.00,

    # 'reg_lambda': 0,
    # 'reg_alpha': 0,
    # 'updater': 'shotgun'
}

close_params = xgb_params.copy()

num_round = 1000
early_stopping_rounds = 10

max_params = xgb_params.copy()
max_params.update({
    'eta': close_params['eta'] * 2
})

xgb_close_model = XGBSplitDatasetModel(
    dataset_descriptor=close_dataset,
    booster_params=close_params,
    num_round=num_round,
    early_stopping_rounds=early_stopping_rounds,
    obj_name='close'
)

xgb_max_model = XGBSplitDatasetModel(
    dataset_descriptor=max_dataset,
    booster_params=max_params,
    num_round=num_round,
    early_stopping_rounds=early_stopping_rounds,
    obj_name='max'
)

xgb_models = [xgb_close_model, xgb_max_model]

xgb_pair_models = {}

for pair in xgb_training_pairs:
    pair_close_model = XGBSplitDatasetModel(
        dataset_descriptor=pair_close_datasets[pair],
        booster_params=close_params,
        num_round=num_round,
        early_stopping_rounds=early_stopping_rounds,
        obj_name='close'
    )

    pair_max_model = XGBSplitDatasetModel(
        dataset_descriptor=pair_max_datasets[pair],
        booster_params=max_params,
        num_round=num_round,
        early_stopping_rounds=early_stopping_rounds,
        obj_name='max'
    )

    xgb_pair_models[pair] = [pair_close_model, pair_max_model]

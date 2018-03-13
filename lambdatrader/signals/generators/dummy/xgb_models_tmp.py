from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets, ValueSets
from lambdatrader.signals.data_analysis.models import XGBSplitDatasetModel

from lambdatrader.signals.generators.factories import Pairs

xgb_training_pairs = Pairs.all_pairs(); interleaved = True
# xgb_training_pairs = Pairs.n_pairs(); interleaved = True
# xgb_training_pairs = ['BTC_ETH']; interleaved = False
# xgb_training_pairs = ['BTC_XMR']; interleaved = False
# xgb_training_pairs = ['BTC_LTC']; interleaved = False
# xgb_training_pairs = ['BTC_XRP']; interleaved = False
# xgb_training_pairs = ['BTC_STR']; interleaved = False
# xgb_training_pairs = ['BTC_RADS']; interleaved = False
# xgb_training_pairs = ['BTC_RIC']; interleaved = False
# xgb_training_pairs = ['BTC_SC']; interleaved = False
# xgb_training_pairs = ['BTC_VIA']; interleaved = False
# xgb_training_pairs = ['BTC_VTC']; interleaved = False


# xgb_split_date_range = SplitDateRanges.january_3_days_test_3_days_val_7_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_20_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_160_days_train()
xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_360_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_500_days_train()
# xgb_split_date_range = SplitDateRanges.january_20_days_test_20_days_val_rest_train()

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


feature_set = FeatureSets.get_all_periods_last_ten_ohlcv_now_delta()

xgb_n_candles = 48
value_set_cavg = DFFeatureSet(features=[CloseAvgReturn(n_candles=xgb_n_candles)])
# value_set_close = ValueSets.close_return_4h()

value_set_max = DFFeatureSet(features=[MaxReturn(n_candles=xgb_n_candles)])

xgb_model_per_pair = True

xgb_c_thr = 0.03
xgb_m_thr = 0.03

cavg_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=xgb_training_pairs,
    feature_set=feature_set,
    value_set=value_set_cavg,
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

pair_cavg_datasets = {}
pair_max_datasets = {}

for pair in xgb_training_pairs:
    pair_cavg_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=[pair],
        feature_set=feature_set,
        value_set=value_set_cavg,
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
    pair_cavg_datasets[pair] = pair_cavg_dataset
    pair_max_datasets[pair] = pair_max_dataset

xgb_params = {
    'silent': 1,
    'booster': 'gbtree',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.05,
    'gamma': 0,
    'max_depth': 2,
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

    # 'lambda': 1,
    # 'alpha': 0,
    # 'updater': 'grow_colmaker,prune',

    # 'sample_type': 'uniform',
    # 'normalize_type': 'tree',
    # 'rate_drop': 0.00,
    # 'one_drop': 0,
    # 'skip_drop': 0.00,

    'reg_lambda': 0,
    'reg_alpha': 0,
    # 'updater': 'shotgun'
}

close_params = xgb_params.copy()

num_round = 100
early_stopping_rounds = 10

max_params = xgb_params.copy()
max_params.update({
    'eta': close_params['eta'] * 2
})

xgb_cavg_model = XGBSplitDatasetModel(
    dataset_descriptor=cavg_dataset,
    booster_params=close_params,
    num_round=num_round,
    early_stopping_rounds=early_stopping_rounds,
    obj_name='cavg'
)

xgb_max_model = XGBSplitDatasetModel(
    dataset_descriptor=max_dataset,
    booster_params=max_params,
    num_round=num_round,
    early_stopping_rounds=early_stopping_rounds,
    obj_name='max'
)

xgb_models = [xgb_cavg_model, xgb_max_model]

xgb_pair_models = {}

for pair in xgb_training_pairs:
    pair_cavg_model = XGBSplitDatasetModel(
        dataset_descriptor=pair_cavg_datasets[pair],
        booster_params=close_params,
        num_round=num_round,
        early_stopping_rounds=early_stopping_rounds,
        obj_name='cavg'
    )

    pair_max_model = XGBSplitDatasetModel(
        dataset_descriptor=pair_max_datasets[pair],
        booster_params=max_params,
        num_round=num_round,
        early_stopping_rounds=early_stopping_rounds,
        obj_name='max'
    )

    xgb_pair_models[pair] = [pair_cavg_model, pair_max_model]
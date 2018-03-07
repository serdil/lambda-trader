from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets, ValueSets
from lambdatrader.signals.data_analysis.models import XGBSplitDatasetModel, RFModel

from lambdatrader.signals.generators.factories import Pairs

# rf_training_pairs = Pairs.all_pairs(); interleaved = True
# rf_training_pairs = Pairs.n_pairs(); interleaved = True
rf_training_pairs = ['BTC_ETH']; interleaved = False
# rf_training_pairs = ['BTC_XMR']; interleaved = False
# rf_training_pairs = ['BTC_LTC']; interleaved = False
# rf_training_pairs = ['BTC_XRP']; interleaved = False
# rf_training_pairs = ['BTC_STR']; interleaved = False
# rf_training_pairs = ['BTC_RADS']; interleaved = False
# rf_training_pairs = ['BTC_RIC']; interleaved = False
# rf_training_pairs = ['BTC_SC']; interleaved = False
# rf_training_pairs = ['BTC_VIA']; interleaved = False
# rf_training_pairs = ['BTC_VTC']; interleaved = False


# split_date_range = SplitDateRanges.january_3_days_test_3_days_val_7_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_20_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_160_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_360_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_500_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_rest_train()

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=7)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=20)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=40)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=60)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=90)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=120)
split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=200)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=500)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=5000)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=20)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=200)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=500)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=60, t=20)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=60, t=60)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=60, t=200)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=200, t=200)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=200, t=500)


# feature_set = FeatureSets.get_all_periods_last_five_ohlcv()
feature_set = FeatureSets.get_all_periods_last_ten_ohlcv()
# feature_set = FeatureSets.get_all_periods_last_n_ohlcv(30)
# feature_set = FeatureSets.get_all_periods_last_n_ohlcv(3)


rf_n_candles = 48
value_set_cavg = DFFeatureSet(features=[CloseAvgReturn(n_candles=rf_n_candles)])
# value_set_close = ValueSets.close_return_4h()

value_set_max = DFFeatureSet(features=[MaxReturn(n_candles=rf_n_candles)])

rf_model_per_pair = True

rf_c_thr = 0.01
rf_m_thr = 0.02

n_estimators = 100
max_depth = 12

cavg_n_estimators = n_estimators
max_n_estimators = n_estimators

cavg_max_depth = max_depth
max_max_depth = max_depth

max_features = 'auto'
# max_features = 0.1

cavg_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=rf_training_pairs,
    feature_set=feature_set,
    value_set=value_set_cavg,
    split_date_range=split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=rf_training_pairs,
    feature_set=feature_set,
    value_set=value_set_max,
    split_date_range=split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

pair_cavg_datasets = {}
pair_max_datasets = {}

for pair in rf_training_pairs:
    pair_cavg_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=[pair],
        feature_set=feature_set,
        value_set=value_set_cavg,
        split_date_range=split_date_range,
        exchanges=(POLONIEX,),
        interleaved=False
    )
    pair_max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=[pair],
        feature_set=feature_set,
        value_set=value_set_max,
        split_date_range=split_date_range,
        exchanges=(POLONIEX,),
        interleaved=False
    )
    pair_cavg_datasets[pair] = pair_cavg_dataset
    pair_max_datasets[pair] = pair_max_dataset

rf_cavg_model = RFModel(
    dataset_descriptor=cavg_dataset,
    n_estimators=cavg_n_estimators,
    max_depth=cavg_max_depth,
    max_features=max_features,
    obj_name='cavg'
)

rf_max_model = RFModel(
    dataset_descriptor=max_dataset,
    n_estimators=max_n_estimators,
    max_depth=max_max_depth,
    max_features=max_features,
    obj_name='max'
)

rf_models = [rf_cavg_model, rf_max_model]

rf_pair_models = {}

for pair in rf_training_pairs:
    pair_cavg_model = RFModel(
        dataset_descriptor=pair_cavg_datasets[pair],
        n_estimators=cavg_n_estimators,
        max_depth=cavg_max_depth,
        max_features=max_features,
        obj_name='cavg'
    )

    pair_max_model = RFModel(
        dataset_descriptor=pair_max_datasets[pair],
        n_estimators=max_n_estimators,
        max_depth=max_max_depth,
        max_features=max_features,
        obj_name='max'
    )

    rf_pair_models[pair] = [pair_cavg_model, pair_max_model]

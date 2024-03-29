import numpy as np
import xgboost as xgb

from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn
from lambdatrader.signals.data_analysis.factories import (
    SplitDatasetDescriptors, FeatureSets, ValueSets, SplitDateRanges,
)
from lambdatrader.signals.data_analysis.learning.dummy.xgb_interleaved.large_data.utils import (
    train_xgb, train_xgb_libsvm_cache, train_xgb_buffer,
)
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output

sdd = SplitDatasetDescriptors

# close_dataset = sdd.sdd_1_close()
# max_dataset = sdd.sdd_1_max()


# close_dataset = sdd.sdd_1_close_mini()
# max_dataset = sdd.sdd_1_max_mini()


# use_saved = True
use_saved = False

saved_type = 'libsvm'
# saved_type = 'buffer'

# plot_tree = True
plot_tree = False

num_round = 100
early_stopping_rounds = 10

# pairs = ['BTC_ETH']; interleaved=False
pairs = None; interleaved=True

split_date_range = SplitDateRanges.january_20_days_test_20_days_val_160_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_360_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_500_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_rest_train()
feature_set = FeatureSets.get_all_periods_last_ten_ohlcv_now_delta()

value_set_close = DFFeatureSet(features=[CloseAvgReturn(n_candles=48)])
# value_set_close = ValueSets.close_return_4h()

value_set_max = ValueSets.max_return_4h()

close_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=pairs,
    feature_set=feature_set,
    value_set=value_set_close,
    split_date_range=split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=pairs,
    feature_set=feature_set,
    value_set=value_set_max,
    split_date_range=split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

params = {
    'silent': 1,
    'booster': 'gbtree',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.1,
    'gamma': 0,
    'max_depth': 6,
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

close_params = params.copy()

max_params = params.copy()
max_params.update({
    'eta': close_params['eta'] * 2
})

if use_saved:
    if saved_type == 'libsvm':
        res_close = train_xgb_libsvm_cache(dataset_descriptor=close_dataset,
                                           params=close_params,
                                           num_round=num_round,
                                           early_stopping_rounds=early_stopping_rounds,
                                           obj_name='close')
        res_max = train_xgb_libsvm_cache(dataset_descriptor=max_dataset,
                                         params=max_params,
                                         num_round=num_round,
                                         early_stopping_rounds=early_stopping_rounds,
                                         obj_name='max')
    else:
        res_close = train_xgb_buffer(dataset_descriptor=close_dataset,
                                     params=close_params,
                                     num_round=num_round,
                                     early_stopping_rounds=early_stopping_rounds,
                                     obj_name='close')
        res_max = train_xgb_buffer(dataset_descriptor=max_dataset,
                                   params=max_params,
                                   num_round=num_round,
                                   early_stopping_rounds=early_stopping_rounds,
                                   obj_name='max')
else:
    res_close = train_xgb(dataset_descriptor=close_dataset,
                          params=close_params,
                          num_round=num_round,
                          early_stopping_rounds=early_stopping_rounds,
                          obj_name='close')
    res_max = train_xgb(dataset_descriptor=max_dataset,
                        params=max_params,
                        num_round=num_round,
                        early_stopping_rounds=early_stopping_rounds,
                        obj_name='max')

pred_close, real_close, bst_close = res_close
pred_max, real_max, bst_max = res_max


print()
print('++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++')
print()

pred_min = np.zeros(len(pred_close))
real_min = np.zeros(len(pred_close))

pred_real_close = list(zip(pred_close, real_close))
pred_real_max = list(zip(pred_max, real_max))
pred_real_min = list(zip(pred_min, real_min))

analyze_output(pred_real_close=pred_real_close,
               pred_real_max=pred_real_max,
               pred_real_min=pred_real_min)


if plot_tree:
    import matplotlib.pyplot as plt
    xgb.plot_tree(bst_close, num_trees=0)
    plt.show()

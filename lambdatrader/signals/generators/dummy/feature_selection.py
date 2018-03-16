import random
from pprint import pprint

from math import sqrt

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn, CloseReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets
from lambdatrader.signals.data_analysis.models import BaggingDecisionTreeModel
from lambdatrader.signals.generators.dummy.backtest_util import do_backtest
from lambdatrader.signals.generators.dummy.feature_spaces import (
    all_sampler, ohlcv_sampler,
)
from lambdatrader.signals.generators.dummy.signal_generation import (
    CloseAvgReturnMaxReturnSignalConverter, SignalServer, ModelPredSignalGenerator,
    CloseReturnMaxReturnSignalConverter,
)
from lambdatrader.signals.generators.factories import Pairs


random.seed(0)
# random.seed(1)
# random.seed(2)
# random.seed(3)
# random.seed(4)
# random.seed(5)
# random.seed(6)
# random.seed(7)

# training_pairs = Pairs.all_pairs(); interleaved = True
# training_pairs = Pairs.all_pairs()[:40]; interleaved = True
# training_pairs = Pairs.all_pairs()[:20]; interleaved = True
# training_pairs = Pairs.all_pairs()[:10]; interleaved = True
# training_pairs = Pairs.all_pairs()[25:30]; interleaved = True
# training_pairs = Pairs.all_pairs()[20:25]; interleaved = True
# training_pairs = random.sample(Pairs.all_pairs(), 40); interleaved = True
# training_pairs = random.sample(Pairs.all_pairs(), 20); interleaved = True
# training_pairs = random.sample(Pairs.all_pairs(), 15); interleaved = True
# training_pairs = random.sample(Pairs.all_pairs(), 10); interleaved = True
training_pairs = random.sample(Pairs.all_pairs(), 5); interleaved = True
# training_pairs = random.sample(Pairs.all_pairs(), 1); interleaved = True
# training_pairs = Pairs.n_pairs(); interleaved = True
# training_pairs = ['BTC_ETH']; interleaved = False
# training_pairs = ['BTC_XMR']; interleaved = False
# training_pairs = ['BTC_LTC']; interleaved = False
# training_pairs = ['BTC_XRP']; interleaved = False
# training_pairs = ['BTC_STR']; interleaved = False
# training_pairs = ['BTC_RADS']; interleaved = False
# training_pairs = ['BTC_RIC']; interleaved = False
# training_pairs = ['BTC_SC']; interleaved = False
# training_pairs = ['BTC_VIA']; interleaved = False
# training_pairs = ['BTC_VTC']; interleaved = False
# training_pairs = ['BTC_XCP']; interleaved = False
# training_pairs = ['BTC_XVC']; interleaved = False
# training_pairs = ['BTC_STEEM']; interleaved = False
# training_pairs = ['BTC_ZRX']; interleaved = False

# low success rate
# training_pairs = ['BTC_BTM']; interleaved = False
# training_pairs = ['BTC_VRC']; interleaved = False
# training_pairs = ['BTC_HUC']; interleaved = False
# training_pairs = ['BTC_NXC']; interleaved = False
# training_pairs = ['BTC_NEOS']; interleaved = False
# training_pairs = ['BTC_RIC']; interleaved = False

# few signals
# training_pairs = ['BTC_ETC']; interleaved = False
# training_pairs = ['BTC_PINK']; interleaved = False
# training_pairs = ['BTC_FLO']; interleaved = False
# training_pairs = ['BTC_BCH']; interleaved = False
# training_pairs = ['BTC_GNO']; interleaved = False
# training_pairs = ['BTC_FLDC']; interleaved = False
# training_pairs = ['BTC_NOTE']; interleaved = False

# good
# training_pairs = ['BTC_ZRX']; interleaved = False
# training_pairs = ['BTC_BURST']; interleaved = False

# model_per_pair = True
model_per_pair = False

if model_per_pair:
    n_p = 1
else:
    n_p = len(training_pairs)

# split_date_range = SplitDateRanges.january_3_days_test_3_days_val_7_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_20_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_160_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_360_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_500_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_rest_train()

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=7//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=14//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=20//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=30//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=40//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=60//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=90//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=120//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=200//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=500//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=1000//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=2000//n_p)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=30//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=90//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=200//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=500//n_p)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20//n_p, t=30//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20//n_p, t=60//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20//n_p, t=200//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20//n_p, t=500//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20//n_p, t=1000//n_p)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=50//n_p, t=200//n_p)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=10//n_p, t=30//n_p)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=10, t=120//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=10, t=200//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=10, t=500//n_p)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=10, t=2000//n_p)

split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=10, t=100)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=100)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=200)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=400)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=20, t=800)

fs = FeatureSets

# --------------- saved ------------------------
# seed=0 num_pairs=20 t_v_t=20,20/n_p,200/n_p selection_mode=4 feature_selection_ratio={.9, .7} total_features=20 max_depth=4 max_samples=every_n_candles n_estimators=100
# seed=0 num_pairs=20 t_v_t=20,20/n_p,200/n_p selection_mode=4 feature_selection_ratio={.9, .7} total_features=40 max_depth=4 max_samples=every_n_candles n_estimators=100
# seed=0 num_pairs=20 t_v_t=20,20/n_p,200/n_p selection_mode=4 feature_selection_ratio={.9, .7} total_features=100 max_depth=4 max_samples=every_n_candles n_estimators=100

# seed=0 num_pairs=20 t_v_t=20,20/n_p,500/n_p selection_mode=4 feature_selection_ratio={.9, .7} total_features=20 max_depth=4 max_samples=every_n_candles n_estimators=100


# ? n_estimators=500 +
# ? max_depth=6
# ? max_samples=0.25
# --------------- saved ------------------------

# selection_mode = 0
# selection_mode = 1
# selection_mode = 2
# selection_mode = 3
selection_mode = 4

select_close = True
select_max = True

# feature_sampler = ohlcv_sampler
feature_sampler = all_sampler

feature_selection_ratio = 0.95

feature_selection_target_level = 50

num_total_features = 500

feature_set = feature_sampler.sample(size=num_total_features)

n_candles = 48

value_set_close = DFFeatureSet(features=[CloseAvgReturn(n_candles=n_candles)])
# value_set_close = DFFeatureSet(features=[CloseReturn(n_candles=n_candles)])

value_set_max = DFFeatureSet(features=[MaxReturn(n_candles=n_candles)])

c_thr = 0.01
m_thr = 0.02


one_day_samples = 288
n_samples = (split_date_range.training.end - split_date_range.training.start) \
            // M5.seconds() * n_p
n_samples_sqrt = int(sqrt(n_samples))
samples_every_n_candles = n_samples // n_candles

# max_samples = n_samples_sqrt
# max_samples = 1.00
# max_samples = 0.50
# max_samples = 0.25
# max_samples = 0.10
# max_samples = 0.05
# max_samples = 0.02
# max_samples = 0.01
# max_samples = one_day_samples * 7
# max_samples = one_day_samples * 3
# max_samples = one_day_samples * 1
# max_samples = one_day_samples // 2
# max_samples = one_day_samples // 4
# max_samples = 16384 * 2
# max_samples = 16384
# max_samples = 8192
# max_samples = 4096
# max_samples = 2048
# max_samples = 1024
# max_samples = 512
# max_samples = 256
# max_samples = 128
# max_samples = 64
# max_samples = 16
# max_samples = 8
# max_samples = samples_every_n_candles * 4
max_samples = samples_every_n_candles

# max_depth = 2
# max_depth = 3
max_depth = 4
# max_depth = 6
# max_depth = 8
# max_depth = 10
# max_depth = 12
# max_depth = 16
# max_depth = 20


# n_estimators = max(1024000 // max_samples, 1000)
# # n_estimators = max(512000 // max_samples, 500)
# n_estimators = 20000
# n_estimators = 16000
# n_estimators = 8000
# n_estimators = 4000
# n_estimators = 2000
# n_estimators = 1600
# n_estimators = 1000
# n_estimators = 800
n_estimators = 500
# n_estimators = 400
# n_estimators = 200
# n_estimators = 100
# n_estimators = 50
# n_estimators = 20
# n_estimators = 10

# n_estimators = n_candles * 100
# n_estimators = n_candles * 50
# n_estimators = n_candles * 10
# n_estimators = n_candles * 4
# n_estimators = n_candles * 2
# n_estimators = n_candles

# n_estimators = max(n_samples // max_samples * 10, 500)

# n_estimators = n_samples // max_samples * 200
# n_estimators = n_samples // max_samples * 100
# n_estimators = n_samples // max_samples * 80
# n_estimators = n_samples // max_samples * 60
# n_estimators = n_samples // max_samples * 40
# n_estimators = n_samples // max_samples * 20
# n_estimators = n_samples // max_samples * 10
# n_estimators = n_samples // max_samples * 5
# n_estimators = n_samples // max_samples * 2
# n_estimators = n_samples // max_samples

max_features = 'sqrt'
# max_features = 1.00
# max_features = 0.50
# max_features = 0.30
# max_features = 0.20
# max_features = 0.10
# max_features = 0.05
# max_features = 50
# max_features = 20
# max_features = 10
# max_features = 5

# dt_max_features = 1.00
# dt_max_features = 0.50
# dt_max_features = 0.20
# dt_max_features = 0.10
# dt_max_features = 0.05
dt_max_features = 'sqrt'
# dt_max_features = 'log2'

# oob_score = True
oob_score = False

random_state = 5943923 + 0

print('n_features', feature_set.num_features)
print('params:')

val_set_size = (split_date_range.validation.end - split_date_range.validation.start) // M5.seconds()
train_set_size = (split_date_range.training.end - split_date_range.training.start) // M5.seconds()

pprint({
    'n_candles': n_candles,
    'train_set_size': train_set_size,
    'val_set_size': val_set_size,
    'selection_mode': selection_mode,
    'num_total_features': num_total_features,
    'feature_selection_ratio': feature_selection_ratio,
    'n_features': feature_set.num_features,
    'max_depth': max_depth,
    'dt_max_features': dt_max_features,
    'max_features': max_features,
    'max_samples': max_samples,
    'n_estimators': n_estimators,
    'random_state:': random_state,
})

print('training pairs:', training_pairs)

close_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=training_pairs,
    feature_set=feature_set,
    value_set=value_set_close,
    split_date_range=split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=training_pairs,
    feature_set=feature_set,
    value_set=value_set_max,
    split_date_range=split_date_range,
    exchanges=(POLONIEX,),
    interleaved=interleaved
)

pair_close_datasets = {}
pair_max_datasets = {}

for pair in training_pairs:
    pair_close_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=[pair],
        feature_set=feature_set,
        value_set=value_set_close,
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
    pair_close_datasets[pair] = pair_close_dataset
    pair_max_datasets[pair] = pair_max_dataset

rf_close_model = BaggingDecisionTreeModel(
    dataset_descriptor=close_dataset,
    n_estimators=n_estimators,
    max_samples=max_samples,
    max_features=max_features,
    dt_max_features=dt_max_features,
    max_depth=max_depth,
    random_state=random_state,
    obj_name='close',
    oob_score=oob_score
)

rf_max_model = BaggingDecisionTreeModel(
    dataset_descriptor=max_dataset,
    n_estimators=n_estimators,
    max_samples=max_samples,
    max_features=max_features,
    dt_max_features=dt_max_features,
    max_depth=max_depth,
    random_state=random_state,
    obj_name='max',
    oob_score=oob_score
)


def bprint(string):
    print('======================' + string.upper() + '======================')


def close_model_with_feature_set(feature_set):
    close_datset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=training_pairs, feature_set=feature_set, value_set=value_set_close,
        split_date_range=split_date_range, exchanges=(POLONIEX,), interleaved=interleaved)
    close_model = BaggingDecisionTreeModel(dataset_descriptor=close_datset,
        n_estimators=n_estimators, max_samples=max_samples, max_features=max_features,
        dt_max_features=dt_max_features, max_depth=max_depth, random_state=random_state,
        obj_name='close', oob_score=oob_score)
    return close_model


def max_model_with_feature_set(feature_set):
    max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=training_pairs, feature_set=feature_set, value_set=value_set_max,
            split_date_range=split_date_range, exchanges=(POLONIEX,), interleaved=interleaved)
    max_model = BaggingDecisionTreeModel(dataset_descriptor=max_dataset,
            n_estimators=n_estimators, max_samples=max_samples, max_features=max_features,
            dt_max_features=dt_max_features, max_depth=max_depth, random_state=random_state,
            obj_name='max', oob_score=oob_score)
    return max_model


if selection_mode == 0:
    rf_close_model.train()
    rf_max_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_close_model.select_features_by_ratio(feature_selection_ratio)
        num_new_features = num_total_features - len(selected_features.features)
        new_features = feature_sampler.sample(size=num_new_features)
        selected_feature_set = fs.compose_remove_duplicates(selected_features, new_features)
        print('close round {} num_features:'.format(i), len(selected_feature_set.features))
        rf_close_model = close_model_with_feature_set(selected_feature_set)
        rf_close_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_max_model.select_features_by_ratio(feature_selection_ratio)
        num_new_features = len(selected_features.features)
        new_features = feature_sampler.sample(size=num_new_features)
        max_feature_set = fs.compose_remove_duplicates(selected_features, new_features)
        print('max round {} num_features:'.format(i), len(max_feature_set.features))
        max_model_with_feature_set(max_feature_set)
        rf_max_model.train()
elif selection_mode == 1:
    rf_close_model.train()
    rf_max_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_close_model.select_features_by_ratio(feature_selection_ratio)
        selected_feature_set = fs.compose_remove_duplicates(selected_features)
        print('close round {} num_features:'.format(i), len(selected_feature_set.features))
        rf_close_model = close_model_with_feature_set(selected_feature_set)
        rf_close_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_max_model.select_features_by_ratio(feature_selection_ratio)
        max_feature_set = fs.compose_remove_duplicates(selected_features)
        print('max round {} num_features:'.format(i), len(max_feature_set.features))
        rf_max_model = max_model_with_feature_set(max_feature_set)
        rf_max_model.train()
elif selection_mode == 2:
    rf_close_model.train()
    orig_features = rf_close_model.feature_set
    for i in range(feature_selection_target_level):
        selected_features = rf_close_model.select_features_by_ratio(feature_selection_ratio)
        shrinked_features = orig_features.shrink_to_size(selected_features.num_features)
        print('close round {} num_features:'.format(i), len(selected_features.features))
        rf_close_model = close_model_with_feature_set(selected_features)
        shrinked_close_model = close_model_with_feature_set(shrinked_features)
        print()
        print('selected model:')
        rf_close_model.train()
        print()
        print('shrinked model:')
        shrinked_close_model.train()
elif selection_mode == 3:
    num_features = int(num_total_features / feature_selection_ratio)
    level_features = [[] for _ in range(feature_selection_target_level+1)]
    last_reached_level = 0
    while last_reached_level < feature_selection_target_level:
        for level in range(feature_selection_target_level, -1, -1):
            if float(len(level_features[level])) >= num_features:
                last_reached_level = max(level, last_reached_level)
                bprint('level {}'.format(level))
                features = DFFeatureSet(features=level_features[level][:num_features])
                features_deduped = fs.compose_remove_duplicates(features)
                level_features[level] = level_features[level][num_features:]
                rf_close_model = close_model_with_feature_set(features_deduped)
                rf_close_model.train()
                selected_features = rf_close_model.select_features_by_ratio(feature_selection_ratio)
                level_features[level+1].extend(selected_features.sample())
                break
            elif level == 0:
                bprint('level 0')
                fresh_features = feature_sampler.sample(size=num_features)
                rf_close_model = close_model_with_feature_set(fresh_features)
                rf_close_model.train()
                selected_features = rf_close_model.select_features_by_ratio(feature_selection_ratio)
                level_features[level+1].extend(selected_features.sample())

    for i in range(feature_selection_target_level):
        orig_features = rf_close_model.feature_set
        selected_features = rf_close_model.select_features_by_ratio(feature_selection_ratio)
        print('close round {} num_features:'.format(i), len(selected_features.features))
        rf_close_model = close_model_with_feature_set(selected_features)
        print()
        print('selected model:')
        rf_close_model.train()
elif selection_mode == 4:
    if select_close:
        bprint('[close] initial model')
        rf_close_model.train()
        for i in range(feature_selection_target_level):
            cur_features = rf_close_model.feature_set
            cur_num_features = cur_features.num_features
            num_new_features = int(num_total_features / feature_selection_ratio - cur_num_features)
            new_features = feature_sampler.sample(size=num_new_features)
            combined_features = fs.compose_remove_duplicates(cur_features, new_features)
            rf_close_model = close_model_with_feature_set(combined_features)
            bprint('[close] selection round')
            rf_close_model.train()
            selected_features = rf_close_model.select_features_by_number(num_total_features)
            rf_close_model = close_model_with_feature_set(selected_features)
            bprint('[close] round {} result (num_features: {})'.format(i, selected_features.num_features))
            rf_close_model.train()
    if select_max:
        bprint('[max] initial model')
        rf_max_model.train()
        for i in range(feature_selection_target_level):
            cur_features = rf_max_model.feature_set
            cur_num_features = cur_features.num_features
            num_new_features = int(num_total_features / feature_selection_ratio - cur_num_features)
            new_features = feature_sampler.sample(size=num_new_features)
            combined_features = fs.compose_remove_duplicates(cur_features, new_features)
            rf_max_model = max_model_with_feature_set(combined_features)
            bprint('[max] selection round')
            rf_max_model.train()
            selected_features = rf_max_model.select_features_by_number(num_total_features)
            rf_max_model = max_model_with_feature_set(selected_features)
            bprint('[max] round {} result (num_features: {})'.format(i, selected_features.num_features))
            rf_max_model.train()


models = [rf_close_model, rf_max_model]

pair_models = {}

for pair in training_pairs:
    pair_close_model = BaggingDecisionTreeModel(
        dataset_descriptor=pair_close_datasets[pair],
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        dt_max_features=dt_max_features,
        max_depth=max_depth,
        random_state=random_state,
        obj_name='close',
        oob_score=oob_score
    )

    pair_max_model = BaggingDecisionTreeModel(
        dataset_descriptor=pair_max_datasets[pair],
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        dt_max_features=dt_max_features,
        max_depth=max_depth,
        random_state=random_state,
        obj_name='max',
        oob_score=oob_score
    )

    pair_models[pair] = [pair_close_model, pair_max_model]


# pairs = Pairs.all_pairs()
# pairs = Pairs.n_pairs()
# pairs = Pairs.eth()
# pairs = Pairs.xrp()
# pairs = Pairs.ric()
pairs = training_pairs

start_date = split_date_range.test.start + n_candles * M5.seconds()
end_date = split_date_range.test.end
n_candles = n_candles
model_per_pair = model_per_pair
pair_models = pair_models
models = models
c_thr = c_thr
m_thr = m_thr

signal_converter = CloseAvgReturnMaxReturnSignalConverter(c_thr=c_thr,
                                                          m_thr=m_thr, n_candles=n_candles)

# signal_converter = CloseReturnMaxReturnSignalConverter(c_thr=c_thr,
#                                                        m_thr=m_thr,
#                                                        n_candles=n_candles)
signal_server = SignalServer(models=models,
                             signal_converter=signal_converter,
                             pairs=pairs,
                             pc_start_date=start_date,
                             pc_end_date=end_date,
                             model_per_pair=model_per_pair,
                             pair_models=pair_models)

cs_store = ChunkCachingCandlestickStore.get_for_exchange(POLONIEX)

market_info = BacktestingMarketInfo(candlestick_store=cs_store)

signal_generator = ModelPredSignalGenerator(market_info=market_info,
                                            signal_server=signal_server,
                                            pairs=pairs)


do_backtest(signal_generator, market_info, start_date, end_date)

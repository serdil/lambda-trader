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
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets
from lambdatrader.signals.data_analysis.models import BaggingDecisionTreeModel
from lambdatrader.signals.generators.dummy.backtest_util import do_backtest
from lambdatrader.signals.generators.dummy.feature_spaces import all_samplers_feature_set_sampler
from lambdatrader.signals.generators.dummy.signal_generation import (
    CloseAvgReturnMaxReturnSignalConverter, SignalServer, ModelPredSignalGenerator,
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
# training_pairs = random.sample(Pairs.all_pairs(), 5); interleaved = True
training_pairs = random.sample(Pairs.all_pairs(), 1); interleaved = True
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

split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=50//n_p, t=200//n_p)

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=10//n_p, t=30//n_p)

fs = FeatureSets

nd_5 = fs.get_all_periods_last_n_ohlcv_now_delta(5)
nd_10 = fs.get_all_periods_last_n_ohlcv_now_delta(10)
nd_20 = fs.get_all_periods_last_n_ohlcv_now_delta(20)
nd_30 = fs.get_all_periods_last_n_ohlcv_now_delta(30)
nd_100 = fs.get_all_periods_last_n_ohlcv_now_delta(100)

sd_5 = fs.get_all_periods_last_n_ohlcv_self_delta(5)
sd_10 = fs.get_all_periods_last_n_ohlcv_self_delta(10)
sd_20 = fs.get_all_periods_last_n_ohlcv_self_delta(20)
sd_30 = fs.get_all_periods_last_n_ohlcv_self_delta(30)
sd_100 = fs.get_all_periods_last_n_ohlcv_self_delta(100)

bb_20_1 = fs.get_bbands_timeperiod_last_n(timeperiod=20, n=1)
bb_20_2 = fs.get_bbands_timeperiod_last_n(timeperiod=20, n=2)
bb_20_3 = fs.get_bbands_timeperiod_last_n(timeperiod=20, n=3)
bb_20_5 = fs.get_bbands_timeperiod_last_n(timeperiod=20, n=5)

bb_5_3 = fs.get_bbands_timeperiod_last_n(timeperiod=5, n=3)
bb_7_3 = fs.get_bbands_timeperiod_last_n(timeperiod=7, n=3)
bb_10_3 = fs.get_bbands_timeperiod_last_n(timeperiod=10, n=3)
bb_30_3 = fs.get_bbands_timeperiod_last_n(timeperiod=30, n=3)
bb_40_3 = fs.get_bbands_timeperiod_last_n(timeperiod=40, n=3)
bb_60_3 = fs.get_bbands_timeperiod_last_n(timeperiod=60, n=3)
range_bb_24 = fs.compose_remove_duplicates(*[fs.get_bbands_timeperiod_last_n(i, 1) for i in range(2, 24)])
range_bb_48 = fs.compose_remove_duplicates(*[fs.get_bbands_timeperiod_last_n(i, 1) for i in range(2, 48)])
range_bb_96 = fs.compose_remove_duplicates(*[fs.get_bbands_timeperiod_last_n(i, 1) for i in range(2, 96)])
bb_range_50s5 = fs.compose_remove_duplicates(*[fs.get_bbands_timeperiod_last_n(i, 1) for i in range(2, 50, 5)])

macd_last_3 = fs.get_macd_last_n(3)
macd_last_5 = fs.get_macd_last_n(5)

rsi_last_3 = fs.get_rsi_last_n(3)
rsi_last_5 = fs.get_rsi_last_n(5)
rsi_last_10 = fs.get_rsi_last_n(10)

rsi_7_5 = fs.get_rsi_timeperiod_last_n(7, 5)
rsi_14_5 = fs.get_rsi_timeperiod_last_n(14, 5)
rsi_28_5 = fs.get_rsi_timeperiod_last_n(28, 5)

sma_5_3 = fs.get_sma_timeperiod_self_close_delta_last_n(5, 3)
sma_13_3 = fs.get_sma_timeperiod_self_close_delta_last_n(13, 3)
sma_21_3 = fs.get_sma_timeperiod_self_close_delta_last_n(21, 3)
sma_50_3 = fs.get_sma_timeperiod_self_close_delta_last_n(50, 3)
sma_100_3 = fs.get_sma_timeperiod_self_close_delta_last_n(100, 3)
sma_200_3 = fs.get_sma_timeperiod_self_close_delta_last_n(200, 3)

range_sma_12 = fs.compose_remove_duplicates(*[fs.get_sma_timeperiod_self_close_delta_last_n(i, 1) for i in range(2, 12)])
range_sma_24 = fs.compose_remove_duplicates(*[fs.get_sma_timeperiod_self_close_delta_last_n(i, 1) for i in range(2, 24)])
range_sma_48 = fs.compose_remove_duplicates(*[fs.get_sma_timeperiod_self_close_delta_last_n(i, 1) for i in range(2, 48)])
range_sma_10 = fs.compose_remove_duplicates(*[fs.get_sma_timeperiod_self_close_delta_last_n(i, 1) for i in range(2, 48)])
sma_range_100s5 = fs.compose_remove_duplicates(*[fs.get_sma_timeperiod_self_close_delta_last_n(i, 1) for i in range(2, 100, 5)])

patterns = fs.get_all_candlestick_patterns_last_n()

top_pat_inds = [IndicatorEnum.CDLHIKKAKE, IndicatorEnum.CDLSHORTLINE,
                IndicatorEnum.CDLCLOSINGMARUBOZU, IndicatorEnum.CDLBELTHOLD,
                IndicatorEnum.CDLHIGHWAVE, IndicatorEnum.CDLDOJI, IndicatorEnum.CDLSPINNINGTOP]

hikkake = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLHIKKAKE])
longline = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLLONGLINE])
shortline = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLSHORTLINE])
closingmarubozu = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLCLOSINGMARUBOZU])
belthold = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLBELTHOLD])
highwave = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLHIGHWAVE])
doji = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLDOJI])
spinningtop = fs.get_candlestick_patterns_last_n([IndicatorEnum.CDLSPINNINGTOP])

top_patterns = fs.compose_remove_duplicates(hikkake, longline, shortline, closingmarubozu,
                                            belthold, highwave, doji, spinningtop)

# feature_set = nd_100
# feature_set = nd_30
# feature_set = nd_20
# feature_set = nd_10
# feature_set = nd_5
# feature_set = sd_100
# feature_set = sd_30
# feature_set = sd_20
# feature_set = sd_10
# feature_set = sd_5
# feature_set = fs.compose_remove_duplicates(nd_5, sd_5)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10)
# feature_set = fs.compose_remove_duplicates(nd_20, sd_20)
# feature_set = fs.compose_remove_duplicates(nd_100, sd_100)

# feature_set = fs.compose_remove_duplicates(nd_10, sd_10)

# feature_set = bb_5_3
# feature_set = bb_7_3
# feature_set = bb_10_3
# feature_set = bb_20_3
# feature_set = bb_30_3
# feature_set = range_bb_24
# feature_set = range_bb_48
# feature_set = bb_range_50s5
# feature_set = fs.compose_remove_duplicates(bb_5_3, bb_7_3)
# feature_set = fs.compose_remove_duplicates(bb_5_3, bb_10_3)
# feature_set = fs.compose_remove_duplicates(bb_5_3, bb_20_3)
# feature_set = fs.compose_remove_duplicates(bb_5_3, bb_10_3, bb_20_3)
# feature_set = fs.compose_remove_duplicates(bb_5_3, bb_10_3, bb_20_3, bb_30_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_range_50s5)

# feature_set = rsi_7_5
# feature_set = rsi_14_5
# feature_set = rsi_28_5
# feature_set = fs.compose_remove_duplicates(rsi_7_5, rsi_14_5)
# feature_set = fs.compose_remove_duplicates(rsi_7_5, rsi_14_5, rsi_28_5)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, rsi_14_5)

# feature_set = macd_last_3
# feature_set = macd_last_5

# feature_set = sma_5_3
# feature_set = sma_13_3
# feature_set = sma_21_3
# feature_set = sma_50_3
# feature_set = sma_100_3
# feature_set = sma_200_3
# feature_set = range_sma_12
# feature_set = range_sma_24
# feature_set = range_sma_48
# feature_set = sma_range_100s5
# feature_set = fs.compose_remove_duplicates(sma_5_3, sma_13_3)
# feature_set = fs.compose_remove_duplicates(sma_5_3, sma_13_3, sma_21_3)
# feature_set = fs.compose_remove_duplicates(sma_5_3, sma_13_3, sma_21_3, sma_50_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, sma_5_3, sma_13_3, sma_21_3, sma_50_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, range_sma_48)

# feature_set = hikkake
# feature_set = longline
# feature_set = shortline
# feature_set = closingmarubozu
# feature_set = belthold
# feature_set = highwave
# feature_set = doji
# feature_set = spinningtop
# feature_set = top_patterns
# feature_set = patterns
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, top_patterns)

# feature_set = fs.compose_remove_duplicates(nd_10, sd_10)
# feature_set = fs.compose_remove_duplicates(bb_5_3, bb_20_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3, sma_5_3, sma_13_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3, sma_5_3, sma_13_3, sma_21_3, sma_50_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3, sma_5_3, sma_13_3, sma_21_3, sma_50_3, sma_100_3, sma_200_3)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3, sma_5_3, sma_13_3, sma_21_3, sma_50_3, sma_100_3, sma_200_3, top_patterns)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3, bb_range_50s5, sma_5_3, sma_13_3, sma_21_3, sma_50_3, sma_100_3, sma_200_3, range_sma_48, top_patterns)
# feature_set = fs.compose_remove_duplicates(nd_10, sd_10, bb_5_3, bb_20_3, sma_5_3, sma_13_3, sma_21_3, sma_50_3, sma_100_3, sma_200_3, range_sma_48, top_patterns)

# selection_mode = 0
# selection_mode = 1
# selection_mode = 2
# selection_mode = 3
selection_mode = 4

feature_selection_ratio = 0.90

feature_selection_target_level = 5

num_total_features = 100

feature_set = all_samplers_feature_set_sampler.sample(size=num_total_features)

n_candles = 48

value_set_cavg = DFFeatureSet(features=[CloseAvgReturn(n_candles=n_candles)])

value_set_max = DFFeatureSet(features=[MaxReturn(n_candles=n_candles)])

c_thr = 0.02
m_thr = 0.02


one_day_samples = 288
n_samples = (split_date_range.training.end - split_date_range.training.start) \
            // M5.seconds() * n_p
n_samples_sqrt = int(sqrt(n_samples))
samples_every_n_candles = n_samples // n_candles

# max_samples = n_samples_sqrt
# max_samples = 1.00
# max_samples = 0.50
max_samples = 0.25
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
# max_samples = samples_every_n_candles

# max_depth = 2
# max_depth = 3
# max_depth = 4
max_depth = 6
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
n_estimators = 1000
# n_estimators = 800
# n_estimators = 500
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

print('n_features', len(feature_set.features))
print('params:')
pprint({
    'dt_max_features': dt_max_features,
    'max_features': max_features,
    'max_samples': max_samples,
    'n_est': n_estimators,
    'random_state:': random_state,
})

print('training pairs:', training_pairs)

cavg_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
    pairs=training_pairs,
    feature_set=feature_set,
    value_set=value_set_cavg,
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

pair_cavg_datasets = {}
pair_max_datasets = {}

for pair in training_pairs:
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

rf_cavg_model = BaggingDecisionTreeModel(
    dataset_descriptor=cavg_dataset,
    n_estimators=n_estimators,
    max_samples=max_samples,
    max_features=max_features,
    dt_max_features=dt_max_features,
    max_depth=max_depth,
    random_state=random_state,
    obj_name='cavg',
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

feature_set_sampler = all_samplers_feature_set_sampler


def bprint(string):
    print('======================' + string.upper() + '======================')


def cavg_model_with_feature_set(feature_set):
    cavg_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
        pairs=training_pairs, feature_set=feature_set, value_set=value_set_cavg,
        split_date_range=split_date_range, exchanges=(POLONIEX,), interleaved=interleaved)
    cavg_model = BaggingDecisionTreeModel(dataset_descriptor=cavg_dataset,
        n_estimators=n_estimators, max_samples=max_samples, max_features=max_features,
        dt_max_features=dt_max_features, max_depth=max_depth, random_state=random_state,
        obj_name='cavg', oob_score=oob_score)
    return cavg_model


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
    rf_cavg_model.train()
    rf_max_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_cavg_model.select_features_by_ratio(feature_selection_ratio)
        num_new_features = num_total_features - len(selected_features.features)
        new_features = feature_set_sampler.sample(size=num_new_features)
        selected_feature_set = fs.compose_remove_duplicates(selected_features, new_features)
        print('cavg round {} num_features:'.format(i), len(selected_feature_set.features))
        rf_cavg_model = cavg_model_with_feature_set(selected_feature_set)
        rf_cavg_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_max_model.select_features_by_ratio(feature_selection_ratio)
        num_new_features = len(selected_features.features)
        new_features = feature_set_sampler.sample(size=num_new_features)
        max_feature_set = fs.compose_remove_duplicates(selected_features, new_features)
        print('max round {} num_features:'.format(i), len(max_feature_set.features))
        max_model_with_feature_set(max_feature_set)
        rf_max_model.train()
elif selection_mode == 1:
    rf_cavg_model.train()
    rf_max_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_cavg_model.select_features_by_ratio(feature_selection_ratio)
        selected_feature_set = fs.compose_remove_duplicates(selected_features)
        print('cavg round {} num_features:'.format(i), len(selected_feature_set.features))
        rf_cavg_model = cavg_model_with_feature_set(selected_feature_set)
        rf_cavg_model.train()
    for i in range(feature_selection_target_level):
        selected_features = rf_max_model.select_features_by_ratio(feature_selection_ratio)
        max_feature_set = fs.compose_remove_duplicates(selected_features)
        print('max round {} num_features:'.format(i), len(max_feature_set.features))
        rf_max_model = max_model_with_feature_set(max_feature_set)
        rf_max_model.train()
elif selection_mode == 2:
    rf_cavg_model.train()
    orig_features = rf_cavg_model.feature_set
    for i in range(feature_selection_target_level):
        selected_features = rf_cavg_model.select_features_by_ratio(feature_selection_ratio)
        shrinked_features = orig_features.shrink_to_size(selected_features.num_features)
        print('cavg round {} num_features:'.format(i), len(selected_features.features))
        rf_cavg_model = cavg_model_with_feature_set(selected_features)
        shrinked_cavg_model = cavg_model_with_feature_set(shrinked_features)
        print()
        print('selected model:')
        rf_cavg_model.train()
        print()
        print('shrinked model:')
        shrinked_cavg_model.train()
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
                rf_cavg_model = cavg_model_with_feature_set(features_deduped)
                rf_cavg_model.train()
                selected_features = rf_cavg_model.select_features_by_ratio(feature_selection_ratio)
                level_features[level+1].extend(selected_features.sample())
                break
            elif level == 0:
                bprint('level 0')
                fresh_features = feature_set_sampler.sample(size=num_features)
                rf_cavg_model = cavg_model_with_feature_set(fresh_features)
                rf_cavg_model.train()
                selected_features = rf_cavg_model.select_features_by_ratio(feature_selection_ratio)
                level_features[level+1].extend(selected_features.sample())

    for i in range(feature_selection_target_level):
        orig_features = rf_cavg_model.feature_set
        selected_features = rf_cavg_model.select_features_by_ratio(feature_selection_ratio)
        print('cavg round {} num_features:'.format(i), len(selected_features.features))
        rf_cavg_model = cavg_model_with_feature_set(selected_features)
        print()
        print('selected model:')
        rf_cavg_model.train()
elif selection_mode == 4:
    bprint('[cavg] initial model')
    rf_cavg_model.train()
    for i in range(feature_selection_target_level):
        cur_features = rf_cavg_model.feature_set
        cur_num_features = cur_features.num_features
        num_new_features = int(num_total_features / feature_selection_ratio - cur_num_features)
        new_features = feature_set_sampler.sample(size=num_new_features)
        combined_features = fs.compose_remove_duplicates(cur_features, new_features)
        rf_cavg_model = cavg_model_with_feature_set(combined_features)
        bprint('[cavg] selection round')
        rf_cavg_model.train()
        selected_features = rf_cavg_model.select_features_by_ratio(feature_selection_ratio)
        rf_cavg_model = cavg_model_with_feature_set(selected_features)
        bprint('[cavg] round {} result (num_features: {})'.format(i, selected_features.num_features))
        rf_cavg_model.train()
    bprint('[max] initial model')
    rf_max_model.train()
    for i in range(feature_selection_target_level):
        cur_features = rf_max_model.feature_set
        cur_num_features = cur_features.num_features
        num_new_features = int(num_total_features / feature_selection_ratio - cur_num_features)
        new_features = feature_set_sampler.sample(size=num_new_features)
        combined_features = fs.compose_remove_duplicates(cur_features, new_features)
        rf_max_model = max_model_with_feature_set(combined_features)
        bprint('[max] selection round')
        rf_max_model.train()
        selected_features = rf_max_model.select_features_by_ratio(feature_selection_ratio)
        rf_max_model = max_model_with_feature_set(selected_features)
        bprint('[max] round {} result (num_features: {})'.format(i, selected_features.num_features))
        rf_max_model.train()


models = [rf_cavg_model, rf_max_model]

pair_models = {}

for pair in training_pairs:
    pair_cavg_model = BaggingDecisionTreeModel(
        dataset_descriptor=pair_cavg_datasets[pair],
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        dt_max_features=dt_max_features,
        max_depth=max_depth,
        random_state=random_state,
        obj_name='cavg',
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

    pair_models[pair] = [pair_cavg_model, pair_max_model]


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
                                                          m_thr=m_thr,
                                                          n_candles=n_candles)
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

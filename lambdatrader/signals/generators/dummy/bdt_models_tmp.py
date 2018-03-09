import random
from pprint import pprint

from math import sqrt

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets
from lambdatrader.signals.data_analysis.models import BaggingDecisionTreeModel
from lambdatrader.signals.generators.dummy.backtest_util import do_backtest
from lambdatrader.signals.generators.dummy.signal_generation import (
    CloseAvgReturnMaxReturnSignalConverter, SignalServer, ModelPredSignalGenerator,
)
from lambdatrader.signals.generators.factories import Pairs

# training_pairs = Pairs.all_pairs(); interleaved = True
# training_pairs = Pairs.all_pairs()[:40]; interleaved = True
# training_pairs = Pairs.all_pairs()[:20]; interleaved = True
# training_pairs = Pairs.all_pairs()[:10]; interleaved = True
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


# model_per_pair = True
model_per_pair = False

# split_date_range = SplitDateRanges.january_3_days_test_3_days_val_7_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_20_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_160_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_360_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_500_days_train()
# split_date_range = SplitDateRanges.january_20_days_test_20_days_val_rest_train()

# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=7)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=14)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=20)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=30)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=40)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=60)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=90)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=120)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=200)
split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=500)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=1000)
# split_date_range = SplitDateRanges.jan_n_days_test_m_days_val_k_days_train(20, v=0, t=2000)

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


feature_set = FeatureSets.get_all_periods_last_five_ohlcv()
# feature_set = FeatureSets.get_all_periods_last_ten_ohlcv()
# feature_set = FeatureSets.get_all_periods_last_n_ohlcv(30)
# feature_set = FeatureSets.get_all_periods_last_n_ohlcv(3)


n_candles = 48

value_set_cavg = DFFeatureSet(features=[CloseAvgReturn(n_candles=n_candles)])

value_set_max = DFFeatureSet(features=[MaxReturn(n_candles=n_candles)])

c_thr = 0.02
m_thr = 0.02


one_day_samples = 288
n_samples = (split_date_range.training.end - split_date_range.training.start) \
            // M5.seconds() * len(training_pairs)
n_samples_sqrt = int(sqrt(n_samples))
samples_every_n_candles = n_samples // n_candles

# max_samples = n_samples_sqrt
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
# max_samples = 16384
# max_samples = 8192
# max_samples = 4096
# max_samples = 2048
max_samples = 1024
# max_samples = 512
# max_samples = 256
# max_samples = 128
# max_samples = 64
# max_samples = samples_every_n_candles


# n_estimators = 20000
# n_estimators = 16000
# n_estimators = 8000
# n_estimators = 4000
# n_estimators = 2000
# n_estimators = 1600
# n_estimators = 1000
# n_estimators = 800
# n_estimators = 500
# n_estimators = 400
# n_estimators = 200
# n_estimators = 100
# n_estimators = 20
# n_estimators = n_candles * 100
# n_estimators = n_candles * 50
# n_estimators = n_candles * 10
# n_estimators = n_candles * 4
# n_estimators = n_candles * 2
# n_estimators = n_candles
# n_estimators = n_samples // max_samples * 100
# n_estimators = n_samples // max_samples * 80
# n_estimators = n_samples // max_samples * 60
# n_estimators = n_samples // max_samples * 40
# n_estimators = n_samples // max_samples * 20
n_estimators = n_samples // max_samples * 10
# n_estimators = n_samples // max_samples * 5
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

print('params:')
pprint({
    'dt_max_features': dt_max_features,
    'max_features': max_features,
    'max_samp_days': max_samples / one_day_samples,
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
    random_state=random_state,
    obj_name='max',
    oob_score=oob_score
)

models = [rf_cavg_model, rf_max_model]

pair_models = {}

for pair in training_pairs:
    pair_cavg_model = BaggingDecisionTreeModel(
        dataset_descriptor=pair_cavg_datasets[pair],
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        dt_max_features=dt_max_features,
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

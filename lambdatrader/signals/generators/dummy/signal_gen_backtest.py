from pprint import pprint

from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.config import (
    BACKTESTING_STRATEGIES,
)
from lambdatrader.constants import M5
from lambdatrader.evaluation.utils import statistics_over_periods, period_statistics
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.signals.generators.dummy.signal_generation import (
    SignalServer, CloseAvgReturnMaxReturnSignalConverter, ModelPredSignalGenerator,
    CloseReturnMaxReturnSignalConverter,
)
from lambdatrader.signals.generators.dummy.xgb_models_tmp import (
    xgb_split_date_range, xgb_n_candles, xgb_training_pairs, xgb_model_per_pair, xgb_pair_models,
    xgb_models, xgb_c_thr, xgb_m_thr,
)
from lambdatrader.signals.generators.factories import Pairs


def print_descriptors(signal_generators):
    print()
    print('Descriptors:')
    for signal_generator in signal_generators:
        try:
            pprint(signal_generator.get_algo_descriptor())
        except AttributeError:
            print('Signal generator has no descriptor.')

backtest_strategies = BACKTESTING_STRATEGIES

cs_store = ChunkCachingCandlestickStore.get_for_exchange(POLONIEX)

market_info = BacktestingMarketInfo(candlestick_store=cs_store)

account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

# start_date = date_str_to_timestamp('2018-01-10')
# end_date = date_str_to_timestamp('2018-01-30')

# pairs = Pairs.n_pairs()
# pairs = Pairs.all_pairs()
# pairs = Pairs.eth()
# pairs = Pairs.xrp()


pairs = xgb_training_pairs
start_date = xgb_split_date_range.test.start + xgb_n_candles * M5.seconds()
end_date = xgb_split_date_range.test.end
n_candles = xgb_n_candles
model_per_pair = xgb_model_per_pair
pair_models = xgb_pair_models
models = xgb_models
c_thr = xgb_c_thr
m_thr = xgb_m_thr

# start_date = rf_split_date_range.test.start + rf_n_candles * M5.seconds()
# end_date = rf_split_date_range.test.end
# pairs = rf_training_pairs
# n_candles = rf_n_candles
# model_per_pair = rf_model_per_pair
# pair_models = rf_pair_models
# models = rf_models
# c_thr = rf_c_thr
# m_thr = rf_m_thr

signal_converter = CloseAvgReturnMaxReturnSignalConverter(c_thr=c_thr,
                                                          m_thr=m_thr,
                                                          n_candles=n_candles)


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

signal_generator = ModelPredSignalGenerator(market_info=market_info,
                                            signal_server=signal_server,
                                            pairs=pairs)

signal_generators = [signal_generator]


signal_executor = SignalExecutor(market_info=market_info, account=account)

print_descriptors(signal_generators)

backtest.backtest(account=account, market_info=market_info, signal_generators=signal_generators,
                  signal_executor=signal_executor, start=start_date, end=end_date)

print('Signal Generators Used:')

for signal_generator in signal_generators:
    pprint(signal_generator.__class__.__dict__)
    pprint(signal_generator.__dict__)

print()
print('Backtest Complete!')

print()
print('Estimated Balance:', account.get_estimated_balance())

print_descriptors(signal_generators)

trading_info = signal_executor.get_trading_info()

stats = period_statistics(trading_info=trading_info)

print()
print('Statistics over whole trading period:')
pprint(stats)

stats_over_weekly_periods = statistics_over_periods(trading_info=trading_info, period_days=7)

print()
print('Statistics over weekly periods:')
pprint(stats_over_weekly_periods)

stats_over_monthly_periods = statistics_over_periods(trading_info=trading_info, period_days=30)

print()
print('Statistics over monthly periods:')
pprint(stats_over_monthly_periods)

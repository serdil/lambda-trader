from pprint import pprint

from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.config import (
    BACKTESTING_NUM_DAYS, BACKTESTING_END_OFFSET_DAYS, BACKTESTING_STRATEGIES,
)
from lambdatrader.evaluation.utils import statistics_over_periods, period_statistics
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.signals.generators.factories import (
    CMMModelSignalGeneratorFactory, Pairs,
)
from lambdatrader.utilities.utils import date_floor, seconds


def print_descriptors(signal_generators):
    print()
    print('Descriptors:')
    for signal_generator in signal_generators:
        try:
            pprint(signal_generator.get_algo_descriptor())
        except AttributeError:
            print('Signal generator has no descriptor.')

backtest_num_seconds = date_floor(int(seconds(days=BACKTESTING_NUM_DAYS)))
backtest_end_offset_seconds = date_floor(int(seconds(days=BACKTESTING_END_OFFSET_DAYS)))

backtest_strategies = BACKTESTING_STRATEGIES

cs_store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

market_info = BacktestingMarketInfo(candlestick_store=cs_store)

account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

start_date = market_info.get_max_pair_end_time() \
             - backtest_num_seconds - backtest_end_offset_seconds
end_date = market_info.get_max_pair_end_time() \
           - backtest_end_offset_seconds

# pairs = Pairs.n_pairs()
pairs = Pairs.all_pairs()

sig_gen_fact = CMMModelSignalGeneratorFactory(cs_store=cs_store,
                                              market_info=market_info,
                                              pairs=pairs,
                                              precompute=True,
                                              pc_start_date=start_date,
                                              pc_end_date=end_date)

# rf_sig_gen = sig_gen_fact.get_random_forest_n_days_n_estimators(n_days=7)
# xgb_lin_reg_sig_gen = sig_gen_fact.get_xgb_lin_reg_n_days(n_days=7)
xgb_lin_reg_one_model_sig_gen = (sig_gen_fact.
                                 get_xgb_lin_reg_n_days_one_model_max_pred(n_days=200,
                                                                           training_pairs=None,
                                                                           close_thr=0.03,
                                                                           retrain_interval_days=30))

signal_generators = [xgb_lin_reg_one_model_sig_gen]


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

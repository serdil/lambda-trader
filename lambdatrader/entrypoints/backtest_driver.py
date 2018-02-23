from pprint import pprint

from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.config import (
    BACKTESTING_NUM_DAYS, BACKTESTING_END_OFFSET_DAYS, BACKTESTING_STRATEGIES,
)
from lambdatrader.constants import (
    STRATEGY__RETRACEMENT, STRATEGY__DYNAMIC_RETRACEMENT, STRATEGY__LINREG,
)
from lambdatrader.evaluation.utils import statistics_over_periods, period_statistics
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.signals.factories import LinRegSignalGeneratorFactory
from lambdatrader.signals.generators.generators.dynamic_retracement import \
    DynamicRetracementSignalGenerator
from lambdatrader.signals.generators.generators.retracement import RetracementSignalGenerator
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

market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(POLONIEX))

account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

start_date = market_info.get_max_pair_end_time() \
             - backtest_num_seconds - backtest_end_offset_seconds
end_date = market_info.get_max_pair_end_time() \
           - backtest_end_offset_seconds


signal_generators = []

for strategy_name in backtest_strategies:
    if strategy_name == STRATEGY__RETRACEMENT:
        signal_generators.append(RetracementSignalGenerator(market_info=market_info,
                                                            live=False,
                                                            silent=False,
                                                            optimize=False))
    elif strategy_name == STRATEGY__DYNAMIC_RETRACEMENT:
        signal_generators.append(DynamicRetracementSignalGenerator(market_info=market_info,
                                                                   live=False,
                                                                   silent=False,
                                                                   enable_disable=True))
    elif strategy_name == STRATEGY__LINREG:
        lin_reg_sig_gen_factory = LinRegSignalGeneratorFactory(market_info,
                                                               live=False,
                                                               silent=False)
        signal_generators.extend([
            lin_reg_sig_gen_factory.get_excluding_first_conf_lin_reg_signal_generator(),
            lin_reg_sig_gen_factory.get_excluding_second_conf_lin_reg_signal_generator()
        ])
    else:
        raise ValueError('Invalid strategy name: {}'.format(strategy_name))


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

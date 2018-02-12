from pprint import pprint

from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.config import (
    BACKTESTING_NUM_DAYS, BACKTESTING_END_OFFSET_DAYS,
)
from lambdatrader.evaluation.utils import statistics_over_periods, period_statistics
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.signals.generator_factories import LinRegSignalGeneratorFactory
from lambdatrader.signals.generators.dynamic_retracement import DynamicRetracementSignalGenerator
from lambdatrader.utilities.utils import date_floor

ONE_DAY = 24 * 3600

BACKTEST_NUM_SECONDS = date_floor(ONE_DAY * BACKTESTING_NUM_DAYS)
BACKTEST_END_OFFSET_SECONDS = date_floor(ONE_DAY * BACKTESTING_END_OFFSET_DAYS)

market_info = BacktestingMarketInfo(candlestick_store=CandlestickStore.get_for_exchange(POLONIEX))

account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

start_date = market_info.get_max_pair_end_time() \
             - BACKTEST_NUM_SECONDS - BACKTEST_END_OFFSET_SECONDS
end_date = market_info.get_max_pair_end_time() \
           - BACKTEST_END_OFFSET_SECONDS

lin_reg_sig_gen_factory = LinRegSignalGeneratorFactory(market_info, live=False, silent=False)

signal_generators = [
    # lin_reg_sig_gen_factory.get_first_conf_lin_reg_signal_generator(),
    lin_reg_sig_gen_factory.get_second_conf_lin_reg_signal_generator(),
]

signal_executor = SignalExecutor(market_info=market_info, account=account)

print('Descriptor:')
pprint(signal_generators[0].get_algo_descriptor())

backtest.backtest(account=account, market_info=market_info, signal_generators=signal_generators,
                  signal_executor=signal_executor, start=start_date, end=end_date)

print('Signal Generator Used:')
pprint(signal_generators[0].__class__.__dict__)
pprint(signal_generators[0].__dict__)

print()
print('Backtest Complete!')

print()
print('Descriptor:')
pprint(signal_generators[0].get_algo_descriptor())

print()
print('Estimated Balance:', account.get_estimated_balance())

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

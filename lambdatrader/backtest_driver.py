from pprint import pprint

from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.config import BACKTESTING_NUM_DAYS, BACKTESTING_END_OFFSET_DAYS
from lambdatrader.evaluation.utils import statistics_over_periods, period_statistics
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.history.store import CandlestickStore
from lambdatrader.backtesting import backtest
from lambdatrader.signals.signals import RetracementSignalGenerator

ONE_DAY = 24 * 3600

BACKTEST_NUM_DAYS = ONE_DAY * BACKTESTING_NUM_DAYS
BACKTEST_END_OFFSET_DAYS = ONE_DAY * BACKTESTING_END_OFFSET_DAYS

market_info = BacktestingMarketInfo(candlestick_store=CandlestickStore.get_instance())

account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

start_date = market_info.get_max_pair_end_time() - 1 * BACKTEST_NUM_DAYS - 3 * 24 * 3600
end_date = market_info.get_max_pair_end_time() - 0 * BACKTEST_NUM_DAYS - 3 * 24 * 3600

signal_generators = [
        RetracementSignalGenerator(market_info=market_info)
    ]
signal_executor = SignalExecutor(market_info=market_info, account=account)

backtest.backtest(account=account, market_info=market_info, signal_generators=signal_generators,
                  signal_executor=signal_executor, start=start_date, end=end_date)

print()
print('Backtest Complete!')

print()
print('Estimated Balance:', account.get_estimated_balance())
print('Open Orders:', list(account.get_open_orders()))

trading_info = signal_executor.get_trading_info()

print()
print(trading_info)

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

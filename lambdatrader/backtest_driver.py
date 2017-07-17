from pprint import pprint

from backtesting.account import Account
from backtesting.marketinfo import BacktestMarketInfo
from evaluation.utils import statistics_over_periods, period_statistics
from executors.executors import SignalExecutor
from history.store import CandlestickStore

from backtesting import backtest
from signals.signals import SignalGenerator

ONE_DAY = 24 * 3600

market_info = BacktestMarketInfo(candlestick_store=CandlestickStore.get_instance())

account = Account(balances={'BTC': 100})

start_date = market_info.get_max_pair_end_time() - ONE_DAY * 7
end_date = market_info.get_max_pair_end_time()

signal_generator = SignalGenerator(market_info=market_info)
signal_executor = SignalExecutor(market_info=market_info, account=account)

backtest.backtest(account=account, market_info=market_info, signal_generator=signal_generator,
                  signal_executor=signal_executor, start=start_date, end=end_date)

print()
print('Backtest Complete!')

print()
print('Estimated Balance:', account.get_estimated_balance(market_info))
print('Open Orders:', list(account.get_open_orders()))

trading_info = signal_executor.get_trading_info()

print()
print(trading_info)

stats = period_statistics(trading_info=trading_info)

print()
print('Statistics over whole trading period:')
pprint(stats)

stats_over_periods = statistics_over_periods(trading_info=trading_info)

print()
print('Statistics over weekly periods:')
pprint(stats_over_periods)

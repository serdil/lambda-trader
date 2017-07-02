from pprint import pprint

from backtesting.account import Account
from backtesting.marketinfo import BacktestMarketInfo
from evaluation.statistics import Statistics
from history.store import CandlestickStore
from strategy.backtest import BacktestStrategy

from backtesting import backtest

ONE_DAY = 24 * 3600

market_info = BacktestMarketInfo(CandlestickStore.get_instance())

account = Account({'BTC': 100})

start_date = market_info.get_max_pair_end_time() - ONE_DAY * 1
end_date = market_info.get_max_pair_end_time()

strategy = BacktestStrategy()

backtest.backtest(account, market_info, strategy, start=start_date, end=end_date)

print()
print('Backtest Complete!')

print()
print('Estimated Balance:', account.get_estimated_balance(market_info))
print('Open Orders:', list(account.get_open_orders()))

trading_info = strategy.get_trading_info()

print()
print(trading_info)

statistics = Statistics(trading_info)
stats = statistics.calc_stats_for_period(start_date, end_date)

print()
print('Statistics over whole trading period:')
pprint(stats)

periods = []
period_stats = []

for date in range(start_date, end_date, 7 * ONE_DAY):
    periods.append((date, date + 7 * ONE_DAY))

for period in periods:
    period_stats.append(statistics.calc_stats_for_period(period[0], period[1]))

stats_over_periods = statistics.calc_stats_over_periods(period_stats)

print()
print('Statistics over weekly periods:')
pprint(stats_over_periods)

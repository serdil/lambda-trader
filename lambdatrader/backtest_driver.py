from backtesting.account import Account
from backtesting.marketinfo import BacktestMarketInfo
from history.store import CandlestickStore
from strategy.backtest import BacktestStrategy

from backtesting import backtest

ONE_DAY = 24 * 3600

market_info = BacktestMarketInfo(CandlestickStore.get_instance())

account = Account({'BTC': 100})

start_date = market_info.get_max_pair_end_time() - ONE_DAY * 1

strategy = BacktestStrategy()

backtest.backtest(account, market_info, strategy, start=start_date)

print()
print('Backtest Complete!')
print('Estimated Balance:', account.get_estimated_balance(market_info))
print('Open Orders:', list(account.get_open_orders()))

print()
print('Trading Info:')
print(strategy.get_trading_info())

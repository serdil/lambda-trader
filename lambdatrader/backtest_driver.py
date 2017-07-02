from backtesting.account import Account
from backtesting.marketinfo import BacktestMarketInfo
from history.store import CandlestickStore
from strategy.backtest import BacktestStrategy

from backtesting import backtest

market_info = BacktestMarketInfo(CandlestickStore.get_instance())

account = Account({'BTC': 100})

backtest.backtest(account, market_info, BacktestStrategy())

print(account.get_estimated_balance(market_info))
print(list(account.get_open_orders()))

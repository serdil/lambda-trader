import backtest
from account import Account
from pastdata import get_past_market_info
from strategy import Strategy

market_info = get_past_market_info()

account = Account({'BTC': 100})

backtest.backtest(account, market_info, Strategy())

print(account.get_estimated_balance(market_info))
print(list(account.get_open_orders()))

#print(account)
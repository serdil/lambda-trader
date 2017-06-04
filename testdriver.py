
import utils
from account import Account
from currency import Currency
from pastdata import get_past_market_info
from strategy import Strategy

market_info = get_past_market_info()

account = Account({Currency.BTC: 100})

utils.backtest(account, market_info, Strategy())

print(account.get_estimated_balance(market_info))
print(account.get_balance(Currency.BTC))
print(list(account.get_open_orders()))

#print(account)
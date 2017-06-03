from collections import defaultdict

import actors
import utils
from account import Account
from currency import Currency
from pastdata import get_past_market_info

market_info = get_past_market_info()

account = Account({Currency.BTC: 100})

utils.backtest(account, market_info, actors.actor_market_buyer)

print(account.get_estimated_balance(market_info))
print(account.get_balance(Currency.BTC))
print(list(account.get_orders()))
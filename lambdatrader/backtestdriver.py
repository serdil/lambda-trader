from backtesting.account import Account
from backtesting.pastdata import get_past_market_info
from strategy.strategy import Strategy, VolumeStrategy

from backtesting import backtest

market_info = get_past_market_info()

account = Account({'BTC': 100})

backtest.backtest(account, market_info,
                  [VolumeStrategy(1), VolumeStrategy(2),
                   VolumeStrategy(3), VolumeStrategy(4),
                   VolumeStrategy(5), VolumeStrategy(6)]
                  )

print(account.get_estimated_balance(market_info))
print(list(account.get_open_orders()))

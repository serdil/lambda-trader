from time import sleep

from polxdriver import PolxMarketInfo, PolxAccount
from strategy import PolxStrategy

sleep(5)
market_info = PolxMarketInfo()
sleep(5)
account = PolxAccount()
sleep(10)

strategy = PolxStrategy(market_info, account)

print('PolxStrategy running...')
while True:
    strategy.act()
    sleep(10)

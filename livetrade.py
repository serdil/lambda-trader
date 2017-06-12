from time import sleep

from polxmixins import PolxMarketInfo, PolxAccount
from strategy import PolxStrategy

sleep(5)
market_info = PolxMarketInfo()
sleep(5)
account = PolxAccount(market_info)
sleep(10)

strategy = PolxStrategy()

print('PolxStrategy running...')
while True:
    strategy.act(account, market_info)
    sleep(10)

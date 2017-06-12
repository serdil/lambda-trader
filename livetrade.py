from time import sleep

from polxmixins import PolxMarketInfo, PolxAccount
from strategy import PolxStrategy

sleep(2)
market_info = PolxMarketInfo()
sleep(3)
account = PolxAccount(market_info)
sleep(5)

strategy = PolxStrategy()

while True:
    strategy.act(account, market_info)
    sleep(10)

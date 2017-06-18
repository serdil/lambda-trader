from time import sleep

from loghandlers import get_logger_with_all_handlers
from polxdriver import PolxMarketInfo, PolxAccount
from strategy import PolxStrategy

logger = get_logger_with_all_handlers(__name__)

sleep(5)
market_info = PolxMarketInfo()
sleep(5)
account = PolxAccount()
sleep(10)

strategy = PolxStrategy(market_info, account)

logger.info('PolxStrategy running...')
while True:
    strategy.act()
    sleep(10)

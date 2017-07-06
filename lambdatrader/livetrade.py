from time import sleep

from poloniex import PoloniexError
from strategy.live import PolxStrategy

from loghandlers import get_logger_with_all_handlers
from polx.polxdriver import PolxMarketInfo, PolxAccount

logger = get_logger_with_all_handlers(__name__)

sleep(5)
market_info = PolxMarketInfo()
sleep(5)
account = PolxAccount()
sleep(10)

strategy = PolxStrategy(market_info, account)

logger.info('PolxStrategy running...')
while True:
    try:
        strategy.act()
    except PoloniexError as e:  # TODO convert to own error type
        if str(e).find('Connection timed out.') >= 0:
            logger.warning(str(e))
        else:
            logger.exception('unhandled exception')
    except Exception as e:
        logger.exception('unhandled exception')
    sleep(5)

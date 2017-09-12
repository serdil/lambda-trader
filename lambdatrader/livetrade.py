from time import sleep

from poloniex import PoloniexError

from lambdatrader.strategy.live import PolxStrategy
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.polx.polxdriver import PolxMarketInfo, PolxAccount

logger = get_logger_with_all_handlers(__name__)

sleep(5)
market_info = PolxMarketInfo()
sleep(5)
account = PolxAccount()
sleep(10)

strategy = PolxStrategy(market_info=market_info, account=account)

logger.info('PolxStrategy running...')
while True:
    try:
        strategy.act()
    except PoloniexError as e:  # TODO convert to own error type
        if str(e).find('Connection timed out.') >= 0:
            logger.warning(str(e))
        elif str(e).find('Please do not make more than') >= 0:
            logger.error(str(e))
        elif str(e).find('Invalid json response') >= 0:
            logger.error(str(e))
        else:
            logger.exception('unhandled exception')
    except Exception as e:
        logger.exception('unhandled exception')
    sleep(5)

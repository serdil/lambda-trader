from time import sleep

from lambdatrader.history.store import CandlestickStore
from lambdatrader.polx.marketinfo import PolxMarketInfo
from loghandlers import get_logger_with_all_handlers

logger = get_logger_with_all_handlers(__name__)

market_info = PolxMarketInfo(candlestick_store=CandlestickStore.get_instance(),
                             async_fetch_ticker=False, async_fetch_candlesticks=False)

first_fetch = True

while True:
    try:
        market_info.fetch_candlesticks()
        logger.debug('fetch completed')
        market_info.candlestick_store.persist_chunks()
        logger.debug('persisted chunks')
        if first_fetch:
            logger.info('first fetch completed')
            first_fetch = False
        sleep(1)
    except Exception:
        logger.exception('unhandled exception in sync_polx_candlesticks. retrying in 10 seconds...')
        sleep(10)

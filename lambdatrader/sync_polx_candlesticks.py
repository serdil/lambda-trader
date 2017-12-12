from logging import getLogger
from time import sleep

from history.store import CandlestickStore
from polx.marketinfo import PolxMarketInfo

logger = getLogger(__name__)

market_info = PolxMarketInfo(candlestick_store=CandlestickStore.get_instance(),
                             async_fetch_ticker=False, async_fetch_candlesticks=False)

while True:
    try:
        market_info.fetch_candlesticks()
        sleep(1)
    except Exception:
        logger.exception('unhandled exception in sync_polx_candlesticks. retrying in 10 seconds...')
        sleep(10)

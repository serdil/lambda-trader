import sys
from time import sleep

from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.constants import M5, PeriodEnum
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.exchanges.poloniex.marketinfo import PolxMarketInfo
from lambdatrader.loghandlers import get_logger_with_all_handlers

periods = [M5]

if len(sys.argv) == 2:
    periods = [PeriodEnum.from_name(period_name) for period_name in sys.argv[1].split(',')]

print('periods:',periods)

logger = get_logger_with_all_handlers('sync_polx_candlesticks')

market_info = PolxMarketInfo(candlestick_store=ChunkCachingCandlestickStore.get_for_exchange(POLONIEX),
                             async_fetch_ticker=False, async_fetch_candlesticks=False)

first_fetch = True

while True:
    try:
        for period in periods:
            market_info.fetch_candlesticks(period=period)
            market_info.candlestick_store.persist_chunks()
        logger.debug('fetch completed')
        logger.debug('persisted chunks')
        if first_fetch:
            logger.info('first fetch completed')
            first_fetch = False
        sleep(1)
    except Exception:
        logger.exception('unhandled exception in sync_polx_candlesticks. retrying in 10 seconds...')
        sleep(10)

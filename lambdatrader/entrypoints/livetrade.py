from time import sleep

from poloniex import PoloniexError

from lambdatrader.candlestick_stores.candlestickstore import CandlestickStore
from lambdatrader.config import ASYNC_FETCH_POLX_CANDLESTICKS
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.exchanges.poloniex.account import PolxAccount
from lambdatrader.exchanges.poloniex.marketinfo import PolxMarketInfo
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.signals.generators.linreg import LinRegSignalGenerator

logger = get_logger_with_all_handlers(__name__)

market_info = PolxMarketInfo(candlestick_store=CandlestickStore.get_for_exchange(POLONIEX),
                             async_fetch_ticker=False,
                             async_fetch_candlesticks=ASYNC_FETCH_POLX_CANDLESTICKS)
account = PolxAccount()

sleep(3)

signal_generator = LinRegSignalGenerator(market_info=market_info,
                                         live=True,
                                         silent=False)
signal_executor = SignalExecutor(market_info=market_info, account=account, live=True, silent=False)

logger.info('strategy descriptor: %s', signal_generator.get_algo_descriptor())

logger.info('bot running...')

trade_signals = []

while True:
    try:
        tracked_signals = signal_executor.act(signals=trade_signals)
        trade_signals = signal_generator.generate_signals(tracked_signals=tracked_signals)
        sleep(1)
    except PoloniexError as e:  # TODO convert to own error type
        if str(e).find('Connection timed out.') >= 0:
            logger.warning(str(e))
        elif str(e).find('Please do not make more than') >= 0:
            logger.error(str(e))
        elif str(e).find('Invalid json response') >= 0:
            logger.error(str(e))
        else:
            logger.exception('unhandled exception, waiting for 10 seconds before retrying...')
            sleep(10)
    except Exception as e:
        logger.exception('unhandled exception, waiting for 10 seconds before retrying...')
        sleep(10)

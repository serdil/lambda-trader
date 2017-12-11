from threading import Lock, Thread
from time import sleep

from poloniex import PoloniexError

from lambdatrader.executors.utils import retry_on_exception
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.models.ticker import Ticker
from lambdatrader.polx.polxclient import polo
from lambdatrader.polx.utils import APICallExecutor, map_exception
from lambdatrader.utils import get_now_timestamp
from lambdatrader.marketinfo.marketinfo import BaseMarketInfo
from lambdatrader.models.enums.exchange import ExchangeEnum


class PolxMarketInfo(BaseMarketInfo):

    def on_pair_candlestick(self, handler):
        raise NotImplementedError

    def on_pair_tick(self, handler):
        raise NotImplementedError

    def on_all_pairs_candlestick(self, handler):
        raise NotImplementedError

    def on_all_pairs_tick(self, handler):
        raise NotImplementedError

    def __init__(self):
        self.logger = get_logger_with_all_handlers(__name__)

        self.__ticker = {}
        self.__ticker_lock = Lock()
        self.__start_fetcher_thread()

    def get_exchange(self) -> ExchangeEnum:
        return ExchangeEnum.POLONIEX

    def __start_fetcher_thread(self):
        t = Thread(target=self.fetcher)
        t.start()

    def get_market_date(self):
        return get_now_timestamp()

    def get_pair_candlestick(self, pair, ind=0):
        raise NotImplementedError()

    def get_pair_latest_candlestick(self, pair):
        raise NotImplementedError()

    def get_pair_ticker(self, pair):
        return self.__ticker[pair]

    def get_pair_last_24h_btc_volume(self, pair):
        return self.__ticker[pair].base_volume

    def get_pair_last_24h_high(self, pair):
        self.lock_ticker()
        value = self.__ticker[pair].high24h
        self.unlock_ticker()
        return value

    def get_active_pairs(self):
        self.lock_ticker()
        pairs_list = [pair for pair in self.__ticker if pair[:3] == 'BTC']
        self.unlock_ticker()
        return pairs_list

    def is_candlesticks_supported(self):
        return False

    def fetcher(self):
        self.logger.info('starting to fetch ticker...')
        while True:
            try:
                self.fetch_ticker()
                sleep(2)
            except PoloniexError as e:
                error_string = str(e)
                if error_string.find('Nonce must be greater than') == 0:
                    self.logger.warning(error_string)
                else:
                    self.logger.exception('unhandled exception')
            except Exception as e:
                self.logger.exception('unhandled exception')

    def fetch_ticker(self):
        self.logger.debug('fetching_ticker')
        ticker_response = self.__get_ticker_with_retry()
        ticker_dict = {}
        for currency, info in ticker_response.items():
            ticker_dict[currency] = self.ticker_info_to_ticker(ticker_info=info)
        self.lock_ticker()
        self.__ticker = ticker_dict
        self.unlock_ticker()

    def __get_ticker_with_retry(self):
        return retry_on_exception(
            task=lambda: self.__api_call(call=lambda: polo.returnTicker()),
            logger=self.logger
        )

    @staticmethod
    def ticker_info_to_ticker(ticker_info):
        last = float(ticker_info['last'])
        lowest_ask = float(ticker_info['lowestAsk'])
        highest_bid = float(ticker_info['highestBid'])
        base_volume = float(ticker_info['baseVolume'])
        quote_volume = float(ticker_info['quoteVolume'])
        percent_change = float(ticker_info['percentChange'])
        high24h = float(ticker_info['high24hr'])
        low24h = float(ticker_info['low24hr'])
        is_frozen = int(ticker_info['isFrozen'])
        _id = int(ticker_info['id'])
        ticker = Ticker(last=last, lowest_ask=lowest_ask, highest_bid=highest_bid,
                        base_volume=base_volume, quote_volume=quote_volume,
                        percent_change=percent_change, high24h=high24h, low24h=low24h,
                        is_frozen=is_frozen, _id=_id)
        return ticker

    def lock_ticker(self):
        self.__ticker_lock.acquire()

    def unlock_ticker(self):
        self.__ticker_lock.release()

    @staticmethod
    def __api_call(call):
        try:
            return APICallExecutor.get_instance().call(call=call)
        except PoloniexError as e:
            raise map_exception(e)

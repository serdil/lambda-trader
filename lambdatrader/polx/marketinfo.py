from threading import Thread, RLock
from time import sleep

from poloniex import PoloniexError

from lambdatrader.executors.utils import retry_on_exception
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.marketinfo.marketinfo import BaseMarketInfo
from lambdatrader.models.enums.exchange import ExchangeEnum
from lambdatrader.models.ticker import Ticker
from lambdatrader.polx.polxclient import polo
from lambdatrader.polx.utils import APICallExecutor, map_exception
from lambdatrader.utils import get_now_timestamp, date_floor
from models.candlestick import Candlestick
from polx.constants import OLDEST_DATE


class PolxMarketInfo(BaseMarketInfo):

    def on_pair_candlestick(self, handler):
        raise NotImplementedError

    def on_pair_tick(self, handler):
        raise NotImplementedError

    def on_all_pairs_candlestick(self, handler):
        raise NotImplementedError

    def on_all_pairs_tick(self, handler):
        raise NotImplementedError

    def __init__(self, candlestick_store, async_fetch_ticker=True, async_fetch_candlesticks=True):
        self.logger = get_logger_with_all_handlers(__name__)

        self.candlestick_store = candlestick_store

        self.__ticker = {}
        self.__ticker_lock = RLock()

        if async_fetch_ticker:
            self.__start_ticker_fetcher_thread()

        if async_fetch_candlesticks:
            self.__start_candlestick_fetcher_thread()

    def get_exchange(self) -> ExchangeEnum:
        return ExchangeEnum.POLONIEX

    def __start_ticker_fetcher_thread(self):
        self.fetch_ticker()
        t = Thread(target=self.ticker_fetcher)
        t.start()

    def __start_candlestick_fetcher_thread(self):
        self.fetch_ticker()

        self.logger.info('synchronizing candlesticks...')

        self.fetch_candlesticks()
        t = Thread(target=self.candlestick_fetcher)
        t.start()

    def get_market_date(self):
        return get_now_timestamp()

    def get_pair_candlestick(self, pair, ind=0):
        date = date_floor(self.get_market_date()) - ind * 300
        return self.candlestick_store.get_candlestick(pair=pair, date=date)

    def get_pair_latest_candlestick(self, pair):
        return self.get_pair_candlestick(pair=pair, ind=0)

    def get_pair_ticker(self, pair):
        return self.__ticker[pair]

    def get_pair_last_24h_btc_volume(self, pair):
        return self.__ticker[pair].base_volume

    def get_pair_last_24h_high(self, pair):
        with self.__ticker_lock:
            return self.__ticker[pair].high24h

    def get_active_pairs(self):
        with self.__ticker_lock:
            return [pair for pair in self.__ticker if pair[:3] == 'BTC']

    def is_candlesticks_supported(self):
        return True

    def ticker_fetcher(self):
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

    def candlestick_fetcher(self):
        self.logger.info('starting to fetch candlesticks...')
        while True:
            try:
                self.fetch_candlesticks()
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
        with self.__ticker_lock:
            self.__ticker = ticker_dict

    def fetch_candlesticks(self):
        self.logger.debug('fetching_candlesticks')
        with self.__ticker_lock:
            if self.__ticker == {}:
                self.fetch_ticker()

        pairs = self.get_active_pairs()

        for pair in pairs:
            self.fetch_pair_candlesticks(pair)

    def fetch_pair_candlesticks(self, pair):
        self.logger.debug('fetching_pair_candlesticks: %s', pair)
        start_date = self.candlestick_store.get_pair_newest_date(pair)
        if start_date is None:
            start_date = OLDEST_DATE

        end_date = self.get_market_date()

        if end_date - start_date > 300:
            candlesticks = self.__get_pair_candlesticks_with_retry(pair, start_date, end_date)
            self.logger.debug('fetched_pair_candlesticks: %s %s', pair, len(candlesticks))
            for candlestick in candlesticks:
                self.candlestick_store.append_candlestick(pair, candlestick)

    def __get_pair_candlesticks_with_retry(self, pair, start_date, end_date):
        return retry_on_exception(
            task=lambda: self.__get_pair_candlesticks(pair, start_date, end_date),
            logger=self.logger
        )

    def __get_pair_candlesticks(self, pair, start_date, end_date):
        polx_chart_data = self.__api_call(
            lambda: polo.returnChartData(currencyPair=pair, period=300,
                                         start=start_date, end=end_date)
        )

        return self.polx_chart_data_to_candlesticks(polx_chart_data)

    @staticmethod
    def polx_chart_data_to_candlesticks(chart_data):
        candlesticks = []
        for c in chart_data:
            candlesticks.append(Candlestick(date=c['date'], high=c['high'],  low=c['low'],
                                            _open=c['open'], close=c['close'],
                                            base_volume=c['volume'], quote_volume=c['quoteVolume'],
                                            weighted_average=c['weightedAverage']))
        return candlesticks

    @staticmethod
    def ticker_btc_pairs(ticker):
        for pair_name in ticker.keys():
            if pair_name[:3] == 'BTC':
                yield pair_name

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

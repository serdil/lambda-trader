from threading import Thread, RLock
from time import sleep

from poloniex import PoloniexError

from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.exchanges.poloniex.constants import OLDEST_DATE
from lambdatrader.exchanges.poloniex.polxclient import polo
from lambdatrader.exchanges.poloniex.utils import APICallExecutor, map_exception
from lambdatrader.executors.utils import retry_on_exception
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.indicators import Indicators
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.marketinfo import BaseMarketInfo
from lambdatrader.models.candlestick import Candlestick
from lambdatrader.models.ticker import Ticker
from lambdatrader.utilities.utils import get_now_timestamp


class PolxMarketInfo(BaseMarketInfo):

    def __init__(self, candlestick_store, async_fetch_ticker=True, async_fetch_candlesticks=True):
        self.logger = get_logger_with_all_handlers(__name__)

        self.candlestick_store = candlestick_store
        self.indicators = Indicators(self)

        self.__ticker = {}
        self.__ticker_lock = RLock()

        if async_fetch_ticker:
            self.__start_ticker_fetcher_thread()

        if async_fetch_candlesticks:
            self.__start_candlestick_fetcher_thread()

    def on_pair_candlestick(self, handler):
        raise NotImplementedError

    def on_pair_tick(self, handler):
        raise NotImplementedError

    def on_all_pairs_candlestick(self, handler):
        raise NotImplementedError

    def on_all_pairs_tick(self, handler):
        raise NotImplementedError

    def get_exchange(self) -> ExchangeEnum:
        return ExchangeEnum.POLONIEX

    def __start_ticker_fetcher_thread(self):
        self.fetch_ticker()
        t = Thread(target=self.__ticker_fetcher)
        t.start()

    def __start_candlestick_fetcher_thread(self):
        self.fetch_ticker()

        self.logger.info('synchronizing candlesticks...')

        self.fetch_candlesticks()
        t = Thread(target=self.__candlestick_fetcher)
        t.start()

    def get_market_date(self):
        return self.market_date

    @property
    def market_date(self):
        return get_now_timestamp()

    def get_pair_candlestick(self, pair, ind=0, period=M5, allow_lookahead=False):
        if period is not M5:
            raise NotImplementedError
        date = self.candlestick_store.get_pair_period_newest_date(pair) - ind * period.seconds()
        return self.candlestick_store.get_candlestick(pair=pair, date=date)

    def get_pair_latest_candlestick(self, pair, period=M5):
        if period is not M5:
            raise NotImplementedError
        return self.get_pair_candlestick(pair=pair, ind=0, period=period)

    def get_pair_ticker(self, pair):
        return self.__ticker[pair]

    def get_pair_last_24h_btc_volume(self, pair):
        return self.__ticker[pair].base_volume

    def get_pair_last_24h_high(self, pair):
        with self.__ticker_lock:
            return self.__ticker[pair].high24h

    def get_active_pairs(self, return_usdt_btc=False):
        with self.__ticker_lock:
            return [pair for pair in self.__ticker
                    if pair[:3] == 'BTC' or (return_usdt_btc and pair == 'USDT_BTC')]

    def is_candlesticks_supported(self):
        return True

    def __ticker_fetcher(self):
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

    def __candlestick_fetcher(self):
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

    def fetch_candlesticks(self, period=M5):
        self.logger.debug('fetching_candlesticks')
        with self.__ticker_lock:
            if self.__ticker == {}:
                self.fetch_ticker()

        pairs = self.get_active_pairs(return_usdt_btc=True)

        for pair in pairs:
            self.__fetch_pair_candlesticks(pair, period=period)

    def get_indicator(self, pair, indicator: IndicatorEnum, args, ind=0, period=M5):
        return self.indicators.compute(pair, indicator, args, ind=ind, period=period)

    def __fetch_pair_candlesticks(self, pair, period):
        self.logger.debug('fetching_pair_candlesticks: %s', pair)
        start_date = self.candlestick_store.get_pair_period_newest_date(pair, period=period)
        if start_date is None:
            start_date = OLDEST_DATE

        end_date = self.market_date

        if end_date - start_date > period.seconds():
            candlesticks = self.__get_pair_candlesticks_with_retry(pair, start_date,
                                                                   end_date, period=period)

            self.logger.debug('fetched_pair_candlesticks: %s %s', pair, len(candlesticks))
            for candlestick in candlesticks:
                self.candlestick_store.append_candlestick(pair, candlestick)

    def __get_pair_candlesticks_with_retry(self, pair, start_date, end_date, period=M5):
        return retry_on_exception(
            task=lambda: self.__get_pair_candlesticks(pair, start_date, end_date, period=period),
            logger=self.logger
        )

    def __get_pair_candlesticks(self, pair, start_date, end_date, period=M5):
        polx_chart_data = self.__api_call(
            lambda: polo.returnChartData(currencyPair=pair, period=period.seconds(),
                                         start=start_date, end=end_date)
        )

        return self.polx_chart_data_to_candlesticks(polx_chart_data, period=period)

    @staticmethod
    def polx_chart_data_to_candlesticks(chart_data, period=M5):
        candlesticks = []
        for c in chart_data:
            candlesticks.append(Candlestick(period=period,
                                            date=int(c['date']),
                                            high=float(c['high']),
                                            low=float(c['low']),
                                            _open=float(c['open']),
                                            close=float(c['close']),
                                            base_volume=float(c['volume']),
                                            quote_volume=float(c['quoteVolume']),
                                            weighted_average=float(c['weightedAverage'])))
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

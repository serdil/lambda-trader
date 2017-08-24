from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Lock
from time import sleep

from poloniex import PoloniexError

from account.account import BaseAccount
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.models.order import OrderType, Order
from lambdatrader.models.ticker import Ticker

from lambdatrader.polx.polxclient import polo
from lambdatrader.utils import pair_from, pair_second, get_now_timestamp
from marketinfo.marketinfo import BaseMarketInfo
from models.enums.exchange import ExchangeEnum


class APICallExecutor:
    class __APICallExecutor:
        def __init__(self):
            self.queued_calls = Queue()
            t = Thread(target=self.__executor)
            t.start()

        def call(self, call):
            result_queue = Queue()
            self.__register(call=call, return_queue=result_queue)
            result = result_queue.get()
            if result[1]:
                raise result[1]
            else:
                return result[0]

        def __register(self, call, return_queue):
            self.queued_calls.put((call, return_queue))

        def __executor(self):
            while True:
                try:
                    _function, return_queue = self.queued_calls.get(timeout=0.1)
                    try:
                        return_queue.put((_function(), None))
                    except Exception as e:
                        return_queue.put((None, e))
                except Empty:
                    pass

    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls.__APICallExecutor()
        return cls.__instance


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

    def is_all_pairs_ticker_cheap(self):
        return True

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
        ticker_response = self.__api_call(call=lambda: polo.returnTicker())
        ticker_dict = {}
        for currency, info in ticker_response.items():
            ticker_dict[currency] = self.ticker_info_to_ticker(ticker_info=info)
        self.lock_ticker()
        self.__ticker = ticker_dict
        self.unlock_ticker()

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
        return APICallExecutor.get_instance().call(call=call)


class UnableToFillException(Exception):
    pass


class PolxAccount(BaseAccount):

    def __init__(self):
        self.logger = get_logger_with_all_handlers(__name__)

    def get_exchange(self):
        return ExchangeEnum.POLONIEX

    def get_maker_fee(self, amount):
        return amount * 0.0015

    def get_taker_fee(self, amount):
        return amount * 0.0025

    def new_order(self, order, fill_or_kill=False):
        self.logger.info('new_order: %s', str(order))
        try:
            order_result = self.__polo_put(order=order, fill_or_kill=fill_or_kill)
            return order_result
        except PoloniexError as e:
            if str(e) == 'Unable to fill order completely.':
                raise UnableToFillException()
            else:
                raise e

    def cancel_order(self, order_number):
        self.logger.info('cancel_order: %d', order_number)
        return self.__api_call(call=lambda: polo.cancelOrder(order_number))

    def get_balances(self):
        self.logger.debug('get_balances')
        balances = self.__api_call(call=lambda: polo.returnBalances())
        return balances

    def get_open_orders(self):
        self.logger.debug('get_open_orders')
        open_orders_response = self.__api_call(call=lambda: polo.returnOpenOrders())
        open_orders = {}
        for key, value in open_orders_response.items():
            for item in value:
                order = self.__order_from_polx_info(key=key, value=item)
                open_orders[order.get_order_number()] = order
        return open_orders

    @staticmethod
    def __order_from_polx_info(key, value):
        currency = pair_second(key)
        _type = OrderType.BUY if value['type'] == 'buy' else OrderType.SELL
        date = datetime.strptime(value['date'], '%Y-%m-%d %H:%M:%S').timestamp()
        order_number = int(value['orderNumber'])
        amount = float(value['amount'])
        price = float(value['rate'])
        return Order(currency=currency, _type=_type, price=price,
                     amount=amount, date=date, order_number=order_number)

    def get_estimated_balance(self, __market_info=None):
        self.logger.debug('get_estimated_balance')
        complete_balances = self.__api_call(call=lambda: polo.returnCompleteBalances())

        estimated_balance = 0.0
        for info in complete_balances.values():
            estimated_balance += float(info['btcValue'])

        return estimated_balance

    def __polo_put(self, order, fill_or_kill=False):
        if order.get_amount() == -1:
            amount = float(self.__api_call(
                call=lambda: polo.returnBalances())[order.get_currency()])
        else:
            amount = order.get_amount()

        if order.get_type() == OrderType.BUY:
            buy_result = self.__api_call(
                lambda: polo.buy(currencyPair=pair_from('BTC', order.get_currency()),
                                 rate=order.get_price(),
                                 amount=amount,
                                 orderType='fillOrKill' if fill_or_kill else False)
            )
            return buy_result
        elif order.get_type() == OrderType.SELL:
            sell_result = self.__api_call(
                lambda: polo.sell(currencyPair=pair_from('BTC', order.get_currency()),
                                  rate=order.get_price(),
                                  amount=amount, 
                                  orderType='fillOrKill' if fill_or_kill else False)
            )
            return sell_result

    def get_balance(self, currency):
        raise NotImplementedError

    def buy(self, currency, price, amount):
        raise NotImplementedError

    def sell(self, currency, price, amount):
        raise NotImplementedError

    def get_order(self, order_number):
        raise NotImplementedError

    def get_open_buy_orders(self):
        raise NotImplementedError

    def get_open_sell_orders(self):
        raise NotImplementedError

    def on_pair_candlestick(self, handler):
        raise NotImplementedError

    def on_pair_tick(self, handler):
        raise NotImplementedError

    def on_all_pairs_candlestick(self, handler):
        raise NotImplementedError

    def on_all_pairs_tick(self, handler):
        raise NotImplementedError

    @staticmethod
    def __api_call(call):
        return APICallExecutor.get_instance().call(call)


from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Lock
from time import sleep

from poloniex import PoloniexError

from order import OrderType, Order
from poloniexclient import polo
from ticker import Ticker
from utils import pair_from, pair_second, get_now_timestamp


class APICallExecutor:
    class __APICallExecutor:
        def __init__(self):
            self.queued_calls = Queue()
            t = Thread(target=self.__executor)
            t.start()

        def call(self, call):
            result_queue = Queue()
            self.__register(call, result_queue)
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
                    function, return_queue = self.queued_calls.get(timeout=0.1)
                    try:
                        return_queue.put((function(), None))
                    except Exception as e:
                        return_queue.put((None, e))
                except Empty:
                    pass


    __instance = __APICallExecutor()

    @classmethod
    def get_instance(cls):
        return cls.__instance


class PolxMarketInfo:

    def __init__(self):
        self.__ticker = {}
        self.__ticker_lock = Lock()
        self.__start_fetcher_thread()

    def __start_fetcher_thread(self):
        t = Thread(target=self.fetcher)
        t.start()

    def get_market_time(self):
        return get_now_timestamp()

    def set_market_time(self, timestamp):
        raise NotImplementedError()

    def get_pair_candlestick(self, currency_pair, ind=0):
        raise NotImplementedError()

    def get_pair_latest_candlestick(self, currency_pair):
        raise NotImplementedError()

    def get_pair_ticker(self, currency_pair):
        return self.__ticker[currency_pair]

    def get_pair_last_24h_btc_volume(self, currency_pair):
        return self.__ticker[currency_pair].base_volume

    def get_pair_last_24h_high(self, currency_pair):
        self.lock_ticker()
        value = self.__ticker[currency_pair].high24h
        self.unlock_ticker()
        return value

    def pairs(self):
        self.lock_ticker()
        pairs_list = [pair for pair in self.__ticker if pair[:3] == 'BTC']
        self.unlock_ticker()
        return pairs_list

    def fetcher(self):
        print('PolxMarketInfo fetching...')
        while True:
            try:
                self.fetch_ticker()
                sleep(2)
            except PoloniexError as e:
                error_string = str(e)
                if error_string.find('Nonce must be greater than') == 0:
                    print(e)
                else:
                    raise e

    def fetch_ticker(self):
        ticker_response = self.__api_call(lambda: polo.returnTicker())
        ticker_dict = {}
        for currency, info in ticker_response.items():
            ticker_dict[currency] = self.ticker_info_to_ticker(info)
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
        id = int(ticker_info['id'])
        ticker = Ticker(last=last, lowest_ask=lowest_ask, highest_bid=highest_bid,
                        base_volume=base_volume, quote_volume=quote_volume,
                        percent_change=percent_change, high24h=high24h, low24h=low24h,
                        is_frozen=is_frozen, id=id)
        return ticker

    def lock_ticker(self):
        self.__ticker_lock.acquire()

    def unlock_ticker(self):
        self.__ticker_lock.release()

    @staticmethod
    def __api_call(call):
        return APICallExecutor.get_instance().call(call)


class UnableToFillException(Exception):
    pass


class PolxAccount:

    def new_order(self, order, fill_or_kill=False):
        try:
            order_result = self.__polo_put(order, fill_or_kill=fill_or_kill)
            return order_result
        except PoloniexError as e:
            if str(e) == 'Unable to fill order completely.':
                raise UnableToFillException()
            else:
                raise e

    def cancel_order(self, order_number):
        return self.__api_call(lambda: polo.cancelOrder(order_number))

    def get_balances(self):
        balances = self.__api_call(lambda: polo.returnBalances())
        return balances

    def get_open_orders(self):
        open_orders_response = self.__api_call(lambda: polo.returnOpenOrders())
        open_orders = {}
        for key, value in open_orders_response.items():
            for item in value:
                order = self.__order_from_polx_info(key, item)
                open_orders[order.get_order_number()] = order
        return open_orders

    @staticmethod
    def __order_from_polx_info(key, value):
        currency = pair_second(key)
        type = OrderType.BUY if value['type'] == 'buy' else OrderType.SELL
        timestamp = datetime.strptime(value['date'], '%Y-%m-%d %H:%M:%S').timestamp()
        order_number = int(value['orderNumber'])
        amount = float(value['amount'])
        price = float(value['rate'])
        return Order(currency, type, price, amount, timestamp, order_number=order_number)

    def get_estimated_balance(self):
        complete_balances = self.__api_call(lambda: polo.returnCompleteBalances())

        estimated_balance = 0.0
        for info in complete_balances.values():
            estimated_balance += float(info['btcValue'])

        return estimated_balance

    def __polo_put(self, order, fill_or_kill=False):
        if order.get_amount() == -1:
            amount = float(self.__api_call(lambda: polo.returnBalances())[order.get_currency()])
        else:
            amount = order.get_amount()

        if order.get_type() == OrderType.BUY:
            buy_result = self.__api_call(lambda: polo.buy(pair_from('BTC', order.get_currency()), order.get_price(),
                                                          amount,
                                                          orderType='fillOrKill' if fill_or_kill else False))
            return buy_result
        elif order.get_type() == OrderType.SELL:
            sell_result = self.__api_call(lambda: polo.sell(pair_from('BTC', order.get_currency()), order.get_price(),
                                                            amount,
                                                            orderType='fillOrKill' if fill_or_kill else False))
            return sell_result

    @staticmethod
    def __api_call(call):
        return APICallExecutor.get_instance().call(call)


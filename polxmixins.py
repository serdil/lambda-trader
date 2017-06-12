from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Lock
from time import sleep

from poloniex import PoloniexError

from order import OrderType, Order
from poloniexclient import polo
from ticker import Ticker
from utils import pair_from, pair_second


class PolxMarketInfo:

    def __init__(self):
        self.__ticker = {}
        self.__ticker_lock = Lock()
        self.__start_fetcher_thread()

    def __start_fetcher_thread(self):
        t = Thread(target=self.fetcher)
        t.start()

    def get_market_time(self):
        return datetime.utcnow().timestamp()

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
        print('PolxMarketInfo starting to fetch')
        while True:
            print('PolxMarketInfo fetching')
            self.fetch_ticker()
            sleep(1)

    def fetch_ticker(self):
        ticker_response = polo.returnTicker()
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


class UnableToFillException(Exception):
    pass


class PolxAccount:

    def __init__(self, market_info):
        self.__market_info = market_info
        self.__transactions = Queue()
        self.__cancel_queue = Queue()
        self.__open_orders = {}
        self.__balances = {}
        self.__complete_balances = {}
        self.__open_orders_lock = Lock()
        self.__balances_lock = Lock()
        self.__complete_balances_lock = Lock()
        self.__transactions_lock = Lock()
        self.__balance_series = []
        self.__start_fetcher_thread()
        self.__start_executor_thread()

    def __start_fetcher_thread(self):
        t = Thread(target=self.fetcher)
        t.start()

    def __start_executor_thread(self):
        t = Thread(target=self.executor)
        t.start()

    def sell(self, currency, amount, market_info):
        price = market_info.get_pair_ticker().highest_bid
        try:
            sell_result = polo.sell(pair_from('BTC', currency), price, amount, orderType='fillOrKill')
            print('sell result:', sell_result)
        except PoloniexError as e:
            if str(e) == 'Unable to fill order completely.':
                print(e)
            else:
                raise e

    def buy(self, currency, amount, market_info):
        price = market_info.get_pair_ticker().lowest_ask
        try:
            buy_result = polo.buy(pair_from('BTC', currency), price, amount, orderType='fillOrKill')
            print('buy result:', buy_result)
        except PoloniexError as e:
            if str(e) == 'Unable to fill order completely.':
                print(e)
            else:
                raise e

    def new_order(self, order):
        self.__transactions.put(order)

    def new_fill_or_kill_transaction(self, orders):
        self.__transactions.put(orders)

    def get_order(self, order_number):
        self.lock_open_orders()
        order = self.__open_orders[order_number]
        self.unlock_open_orders()
        return order

    def cancel_sell_and_sell_now(self, order_number):
        self.__cancel_queue.put(order_number)

    def get_balance(self, currency):
        self.lock_balances()
        balance = float(self.__balances[currency])
        self.unlock_balances()
        return balance

    def get_open_orders(self):
        self.lock_open_orders()
        open_orders = self.__open_orders.copy()
        self.unlock_open_orders()
        return open_orders

    def get_open_and_pending_orders(self):
        open_and_pending_orders = self.get_open_orders()
        self.lock_transactions()
        for transaction in self.__transactions.queue:
            for order in transaction:
                open_and_pending_orders[order.get_order_number()] = order
        self.unlock_transactions()
        return open_and_pending_orders

    def get_estimated_balance(self):
        self.lock_complete_balances()
        complete_balances = self.__complete_balances
        self.unlock_complete_balances()

        estimated_balance = 0.0
        for info in complete_balances.values():
            estimated_balance += float(info['btcValue'])

        return estimated_balance

    def sample_balance(self):
        self.__balance_series.append(self.get_estimated_balance())

    def max_avg_drawback(self):
        total_drawback = 0.0
        num_drawbacks = 0
        max_drawback = 0.0
        max_balance_so_far = 0.0
        for balance in self.__balance_series:
            if balance > max_balance_so_far:
                max_balance_so_far = balance
            if balance < max_balance_so_far:
                num_drawbacks += 1
                current_drawback = (max_balance_so_far - balance) / max_balance_so_far * 100
                total_drawback += current_drawback
                if current_drawback > max_drawback:
                    max_drawback = current_drawback
        return max_drawback, (total_drawback / num_drawbacks if num_drawbacks > 0 else 0.0)

    def fetcher(self):
        print('PolxAccount starting to fetch')
        while True:
            sleep(1)
            self.fetch_open_orders()
            self.fetch_balances()
            self.fetch_complete_balances()

    def fetch_open_orders(self):
        print('PolxAccount fetching open orders')
        open_orders_response = polo.returnOpenOrders()
        open_orders = {}
        for key, value in open_orders_response.items():
            for item in value:
                order = self.order_from_polx_info(key, item)
                open_orders[order.get_order_number()] = order
        self.lock_open_orders()
        self.__open_orders = open_orders
        self.unlock_open_orders()

    @staticmethod
    def order_from_polx_info(key, value):
        currency = pair_second(key)
        type = OrderType.BUY if value['type'] == 'buy' else OrderType.SELL
        timestamp = datetime.strptime(value['date'], '%Y-%m-%d %H:%M:%S').timestamp()
        order_number = int(value['orderNumber'])
        amount = float(value['amount'])
        price = float(value['rate'])
        return Order(currency, type, price, amount, timestamp, order_number=order_number)

    def fetch_balances(self):
        print('PolxAccount fetching balances')
        balances = polo.returnBalances()
        self.lock_balances()
        self.__balances = balances
        self.unlock_balances()

    def fetch_complete_balances(self):
        print('PolxAccount fetching complete balances')
        complete_balances = polo.returnCompleteBalances()
        self.lock_complete_balances()
        self.__complete_balances = complete_balances
        self.unlock_complete_balances()

    # puts and cancels orders
    def executor(self):
        while True:
            sleep(0.1)
            self.execute_transactions()
            self.execute_cancels()

    def execute_transactions(self):
        while True:
            try:
                self.lock_transactions()
                transaction = self.__transactions.get_nowait()
                self.unlock_transactions()
                self.execute_transaction(transaction)
            except Empty:
                self.unlock_transactions()
                break

    def execute_transaction(self, transaction):
        print('Executing transaction')
        try:
            for order in transaction[:-1]:
                    self.put_order_fill_or_kill(order)
            self.put_order(transaction[-1])
        except UnableToFillException:
            pass

    def put_order_fill_or_kill(self, order):
        try:
            self.__polo_put(order, fill_or_kill=True)
        except PoloniexError as e:
            if str(e) == 'Unable to fill order completely.':
                print(e)
                raise UnableToFillException()
            else:
                raise e

    def put_order(self, order):
        print('put order', order)
        self.__polo_put(order)

    def execute_cancels(self):
        try:
            while True:
                self.__cancel_sell_order_sell_now(self.__cancel_queue.get_nowait())
        except Empty:
            pass

    def __cancel_sell_order_sell_now(self, order):
        print('Cancelling order')
        polo.cancelOrder(order.get_order_number())
        price = self.__market_info.get_pair_ticker(pair_from('BTC', order.get_currency())).highest_bid
        sell_order = Order(order.get_currency(), OrderType.SELL, price, order.get_amount(), self.__market_info.get_market_time())
        self.put_order(sell_order)

    def __polo_put(self, order, fill_or_kill=False):

        if order.get_amount() == -1:
            amount = float(polo.returnBalances()[order.get_currency()])
        else:
            amount = order.get_amount()

        if order.get_type() == OrderType.BUY:
            buy_result = polo.buy(pair_from('BTC', order.get_currency()), order.get_price(),
                                  amount,
                                  orderType='fillOrKill' if fill_or_kill else False)
            print('buy result:', buy_result)
        elif order.get_type() == OrderType.SELL:
            sell_result = polo.sell(pair_from('BTC', order.get_currency()), order.get_price(),
                                  amount,
                                  orderType='fillOrKill' if fill_or_kill else False)
            print('sell result:', sell_result)

    def lock_balances(self):
        self.__balances_lock.acquire()

    def unlock_balances(self):
        self.__balances_lock.release()

    def lock_complete_balances(self):
        self.__complete_balances_lock.acquire()

    def unlock_complete_balances(self):
        self.__complete_balances_lock.release()

    def lock_open_orders(self):
        self.__open_orders_lock.acquire()

    def unlock_open_orders(self):
        self.__open_orders_lock.release()

    def lock_transactions(self):
        self.__transactions_lock.acquire()

    def unlock_transactions(self):
        self.__transactions_lock.release()

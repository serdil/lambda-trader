from collections import defaultdict
from typing import List, Dict, Iterable

from illegalorderexception import IllegalOrderException
from marketinfo import BacktestMarketInfo
from models.order import Order, OrderType
from utils import pair_from


class Account:
    def __init__(self, balances: Dict={'BTC': 100}, orders: List[Order]=[]):
        self.__balance_series = []
        self.__balances = defaultdict(int)
        for currency, balance in balances.items():
            self.__balances[currency] = balance
        self.__orders = []
        for order in orders:
            self.__orders.append(order)

    def sell(self, currency, price, amount, market_info):
        if self.__balances[currency] < amount:
            raise IllegalOrderException
        self.__balances[currency] -= amount
        self.__balances['BTC'] += amount * price - self.get_fee(amount * price)
        self.sample_balance(market_info)

    def buy(self, currency, price, amount, market_info):
        if self.__balances['BTC'] < amount * price:
            raise IllegalOrderException
        self.__balances[currency] += amount - self.get_fee(amount)
        self.__balances['BTC'] -= amount * price
        self.sample_balance(market_info)

    def new_order(self, order: Order):
        if order.get_type() == OrderType.SELL:
            if self.__balances[order.get_currency()] < order.get_amount():
                raise IllegalOrderException
            self.__balances[order.get_currency()] -= order.get_amount()
        elif order.get_type() == OrderType.BUY:
            if self.__balances['BTC'] < order.get_amount() * order.get_price():
                raise IllegalOrderException
            self.__balances['BTC'] -= order.get_amount() * order.get_price()
        self.__orders.append(order)

    def get_order(self, order_number):
        for order in self.__orders:
            if order.get_order_number() == order_number:
                return order
        raise KeyError

    def cancel_order(self, order_number):
        order_ind = None
        for i, order in enumerate(self.__orders):
            if order.get_order_number() == order_number:
                order_ind = i
                break
        if order_ind is not None:
            self.reverse_order_effect(self.__orders[order_ind])
            del self.__orders[order_ind]

    def reverse_order_effect(self, order: Order):
        if order.get_type() == OrderType.SELL:
            self.__balances[order.get_currency()] += order.get_amount()
        elif order.get_type() == OrderType.BUY:
            self.__balances['BTC'] += order.get_amount() * order.get_price()

    def get_open_orders(self) -> Iterable[Order]:
        for order in self.__orders:
            if not order.get_is_filled():
                yield order

    def execute_orders(self, market_info: BacktestMarketInfo):
        for order in self.__orders:
            if not order.get_is_filled():
                if self.order_satisfied(order, market_info):
                    self.fill_order(order)
                    self.remove_filled_orders()
                    self.sample_balance(market_info)

    @staticmethod
    def order_satisfied(order: Order, market_info: BacktestMarketInfo):
        candlestick = market_info.get_pair_latest_candlestick(pair_from('BTC', order.get_currency()))
        if order.get_type() == OrderType.SELL:
            return candlestick.high >= order.get_price()
        elif order.get_type() == OrderType.BUY:
            return candlestick.low <= order.get_price()

    def fill_order(self, order: Order):
        print('executing')
        if order.get_type() == OrderType.SELL:
            btc_value = order.get_amount() * order.get_price()
            self.__balances['BTC'] += btc_value - self.get_fee(btc_value)
        elif order.get_type() == OrderType.BUY:
            self.__balances[order.get_currency()] += order.get_amount() - self.get_fee(order.get_amount())
        order.fill()

    def get_fee(self, amount):
        return amount * 0.0025

    def get_balance(self, currency):
        return self.__balances[currency]

    def get_estimated_balance(self, market_info: BacktestMarketInfo):
        estimated_balance = 0
        for currency, balance in self.__balances.items():
            if currency == 'BTC':
                estimated_balance += balance
            else:
                try:
                    candlestick = market_info.get_pair_latest_candlestick(self.pair_from(currency))
                    estimated_balance += balance * candlestick.close
                except KeyError as e:
                    print('KeyError: ', currency, e)
        for order in self.__orders:
            if not order.get_is_filled():
                if order.get_type() == OrderType.SELL:
                    try:
                        candlestick = market_info.get_pair_latest_candlestick(self.pair_from(order.get_currency()))
                        estimated_balance += order.get_amount() * candlestick.close
                    except KeyError as e:
                        print('KeyError: ', order.get_currency(), e)
                elif order.get_type() == OrderType.BUY:
                    estimated_balance += order.get_amount() * order.get_price()
        return estimated_balance

    def remove_filled_orders(self):
        self.__orders = list(filter(lambda order: not order.get_is_filled(), self.__orders))

    def sample_balance(self, market_info):
        self.__balance_series.append(self.get_estimated_balance(market_info))

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

    @staticmethod
    def pair_from(currency):
        return pair_from('BTC', currency)

    def __repr__(self):
        return 'Account(' + str({'balances': self.__balances, 'orders': self.__orders}) + ')'

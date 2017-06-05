from collections import defaultdict
from typing import List, Dict

from currencypair import CurrencyPair
from currency import Currency
from illegalorderexception import IllegalOrderException
from order import Order, OrderType
from marketinfo import MarketInfo


class Account:
    def __init__(self, balances: Dict={Currency.BTC: 100}, orders: List[Order]=[]):
        self.__balances = defaultdict(int)
        for currency, balance in balances.items():
            self.__balances[currency] = balance
        self.__orders = []
        for order in orders:
            self.__orders.append(order)

    def sell(self, currency: Currency, price, amount):
        if self.__balances[currency] < amount:
            raise IllegalOrderException
        self.__balances[currency] -= amount
        self.__balances[Currency.BTC] += amount * price - self.get_fee(amount * price)

    def buy(self, currency, price, amount):
        if self.__balances[Currency.BTC] < amount * price:
            raise IllegalOrderException
        self.__balances[currency] += amount - self.get_fee(amount)
        self.__balances[Currency.BTC] -= amount * price

    def new_order(self, order: Order):
        if order.type == OrderType.SELL:
            if self.__balances[order.currency] < order.amount:
                raise IllegalOrderException
            self.__balances[order.currency] -= order.amount
        elif order.type == OrderType.BUY:
            if self.__balances[Currency.BTC] < order.amount * order.price:
                raise IllegalOrderException
            self.__balances[Currency.BTC] -= order.amount * order.price
        self.__orders.append(order)

    def get_order(self, order_id):
        for order in self.__orders:
            if order.id == order_id:
                return order
        raise KeyError

    def cancel_order(self, order_id):
        order_ind = None
        for i, order in enumerate(self.__orders):
            if order.id == order_id:
                order_ind = i
                break
        if order_ind is not None:
            self.reverse_order_effect(self.__orders[order_ind])
            del self.__orders[order_ind]

    def reverse_order_effect(self, order):
        if order.type == OrderType.SELL:
            self.__balances[order.currency] += order.amount
        elif order.type == OrderType.BUY:
            self.__balances[Currency.BTC] += order.amount * order.price

    def get_open_orders(self):
        for order in self.__orders:
            if not order.is_filled:
                yield order

    def execute_orders(self, market_info: MarketInfo):
        for order in self.__orders:
            if not order.is_filled:
                if self.order_satisfied(order, market_info):
                    self.fill_order(order)
                    self.remove_filled_orders()

    @staticmethod
    def order_satisfied(order: Order, market_info: MarketInfo):
        candlestick = market_info.get_pair_latest_candlestick(CurrencyPair(Currency.BTC, order.currency))
        if order.type == OrderType.SELL:
            return candlestick.high >= order.price
        elif order.type == OrderType.BUY:
            return candlestick.low <= order.price

    def fill_order(self, order):
        print('executing')
        if order.type == OrderType.SELL:
            btc_value = order.amount * order.price
            self.__balances[Currency.BTC] += btc_value - self.get_fee(btc_value)
        elif order.type == OrderType.BUY:
            self.__balances[order.currency] += order.amount - self.get_fee(order.amount)
        order.fill()

    def get_fee(self, amount):
        return amount * 0.0025

    def get_balance(self, currency):
        return self.__balances[currency]

    def get_estimated_balance(self, market_info: MarketInfo):
        estimated_balance = 0
        for currency, balance in self.__balances.items():
            if currency == Currency.BTC:
                estimated_balance += balance
            else:
                try:
                    candlestick = market_info.get_pair_latest_candlestick(self.pair_from(currency))
                    estimated_balance += balance * candlestick.close
                except KeyError as e:
                    print('KeyError: ', currency, e)
        for order in self.__orders:
            if not order.is_filled:
                if order.type == OrderType.SELL:
                    try:
                        candlestick = market_info.get_pair_latest_candlestick(self.pair_from(order.currency))
                        estimated_balance += order.amount * candlestick.close
                    except KeyError as e:
                        print('KeyError: ', order.currency, e)
                elif order.type == OrderType.BUY:
                    estimated_balance += order.amount * order.price
        return estimated_balance

    def remove_filled_orders(self):
        self.__orders = list(filter(lambda order: not order.is_filled, self.__orders))

    @staticmethod
    def pair_from(currency):
        return CurrencyPair(Currency.BTC, currency)

    def __repr__(self):
        return 'Account(' + str({'balances': self.__balances, 'orders': self.__orders}) + ')'

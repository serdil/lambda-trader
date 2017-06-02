from typing import List, Dict

from currencypair import CurrencyPair
from currency import Currency
from illegalorderexception import IllegalOrderException
from order import Order, OrderType
from marketinfo import MarketInfo
from candlestick import Candlestick


class Account:
    def __init__(self, balances: Dict={'BTC': 100}, orders: List(Order)=[]):
        self.balances = {}
        for currency, balance in balances:
            self.balances[currency] = balance
        self.orders = []
        for order in orders:
            self.orders.append(order)

    def sell(self, currency: Currency, price, amount):
        if self.balances[currency] < amount:
            raise IllegalOrderException
        self.balances[currency] -= amount
        self.balances[Currency.BTC] += amount * price


    def buy(self, currency, price, amount):
        if self.balances[Currency.BTC] < amount * price:
            raise IllegalOrderException
        self.balances[currency] += amount
        self.balances[Currency.BTC] -= amount * price

    def new_order(self, order: Order):
        if order.type == OrderType.SELL:
            if self.balances[order.currency] < order.amount:
                raise IllegalOrderException
            self.balances[order.currency] -= order.amount
        elif order.type == OrderType.BUY:
            if self.balances[Currency.BTC] < order.amount * order.price:
                raise IllegalOrderException
            self.balances[Currency.BTC] -= order.amount * order.price
        self.orders.append(order)

    def execute_orders(self, market_info: MarketInfo):
        for order in self.orders:
            if not order.is_filled:
                history = market_info.pairs[CurrencyPair(Currency.BTC, currency)].history

                if self.order_satisfied(order, history):
                    self.fill_order(order)

    @staticmethod
    def order_satisfied(order: Order, history: List(Candlestick)):
        return history[-1].low <= order.price <= history[-1].high

    def fill_order(self, order):
        btc_value = order.amount * order.price
        self.balances[order.currency] += btc_value
        order.fill()
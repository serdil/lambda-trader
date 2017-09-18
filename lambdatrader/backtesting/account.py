from collections import defaultdict
from typing import Dict

from lambdatrader.account.account import BaseAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.models.order import Order
from lambdatrader.models.ordertype import OrderType
from lambdatrader.utils import pair_from
from lambdatrader.models.enums.exchange import ExchangeEnum


class IllegalOrderException(Exception):
    pass


class BacktestingAccount(BaseAccount):
    def get_balances(self):
        pass

    def get_exchange(self):
        return ExchangeEnum.BACKTESTING

    def __init__(self, balances: Dict={'BTC': 100}):
        self.__balances = defaultdict(int)
        for currency, balance in balances.items():
            self.__balances[currency] = balance
        self.__orders = []

    def sell(self, currency, price, amount):
        if self.__balances[currency] < amount:
            raise IllegalOrderException
        self.__balances[currency] -= amount
        self.__balances['BTC'] += amount * price - self.get_taker_fee(amount=amount * price)

    def buy(self, currency, price, amount):
        if self.__balances['BTC'] < amount * price:
            raise IllegalOrderException
        self.__balances[currency] += amount - self.get_taker_fee(amount=amount)
        self.__balances['BTC'] -= amount * price

    def new_order(self, order: Order, __fill_or_kill=None):
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
            self.__reverse_order_effect(order=self.__orders[order_ind])
            del self.__orders[order_ind]

    def __reverse_order_effect(self, order: Order):
        if order.get_type() == OrderType.SELL:
            self.__balances[order.get_currency()] += order.get_amount()
        elif order.get_type() == OrderType.BUY:
            self.__balances['BTC'] += order.get_amount() * order.get_price()

    def get_open_orders(self):
        open_orders = {}
        for order in self.__orders:
            if not order.get_is_filled():
                open_orders[order.get_order_number()] = order
        return open_orders

    def get_open_sell_orders(self):
        open_orders = self.get_open_orders()
        open_sell_orders = {}
        for order_number, order in open_orders.items():
            if order.get_type() == OrderType.SELL:
                open_sell_orders[order_number] = order
        return open_sell_orders

    def get_open_buy_orders(self):
        open_orders = self.get_open_orders()
        open_buy_orders = {}
        for order_number, order in open_orders.items():
            if order.get_type() == OrderType.BUY:
                open_buy_orders[order_number] = order
        return open_buy_orders

    def execute_orders(self, market_info: BacktestingMarketInfo):
        for order in self.__orders:
            if not order.get_is_filled():
                if self.order_satisfied(order=order, market_info=market_info):
                    self.fill_order(order=order)
                    self.__remove_filled_orders()

    @staticmethod
    def order_satisfied(order: Order, market_info: BacktestingMarketInfo):
        candlestick = market_info.get_pair_latest_candlestick(
            pair_from('BTC', order.get_currency())
        )
        if order.get_type() == OrderType.SELL:
            return candlestick.high >= order.get_price()
        elif order.get_type() == OrderType.BUY:
            return candlestick.low <= order.get_price()

    def fill_order(self, order: Order):
        if order.get_type() == OrderType.SELL:
            btc_value = order.get_amount() * order.get_price()
            self.__balances['BTC'] += btc_value - self.get_maker_fee(amount=btc_value)
        elif order.get_type() == OrderType.BUY:
            balance_addition = order.get_amount() - self.get_maker_fee(order.get_amount())
            self.__balances[order.get_currency()] += balance_addition
        order.fill()

    def get_taker_fee(self, amount):
        return amount * 0.0025

    def get_maker_fee(self, amount):
        return amount * 0.0015

    def get_balance(self, currency):
        return self.__balances[currency]

    def get_estimated_balance(self, market_info: BacktestingMarketInfo):
        estimated_balance = 0
        for currency, balance in self.__balances.items():
            if currency == 'BTC':
                estimated_balance += balance
            else:
                try:
                    candlestick = market_info.get_pair_latest_candlestick(
                                                                pair_from('BTC', currency))
                    estimated_balance += balance * candlestick.close
                except KeyError as e:
                    print('KeyError: ', currency, e)
        for order in self.__orders:
            if not order.get_is_filled():
                if order.get_type() == OrderType.SELL:
                    try:
                        candlestick = market_info.get_pair_latest_candlestick(
                            pair_from('BTC', order.get_currency())
                        )
                        estimated_balance += order.get_amount() * candlestick.close
                    except KeyError as e:
                        print('KeyError: ', order.get_currency(), e)
                elif order.get_type() == OrderType.BUY:
                    estimated_balance += order.get_amount() * order.get_price()
        return estimated_balance

    def __remove_filled_orders(self):
        self.__orders = list(filter(lambda order: not order.get_is_filled(), self.__orders))

    def __repr__(self):
        return 'Account(balances={}, orders={})'.format(self.__balances, self.__orders)

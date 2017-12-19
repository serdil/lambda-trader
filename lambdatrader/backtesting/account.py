from collections import defaultdict

from lambdatrader.account import BaseAccount
from lambdatrader.account import NotEnoughBalance, UnableToFillImmediately
from lambdatrader.config import BACKTESTING_TAKER_FEE, BACKTESTING_MAKER_FEE
from lambdatrader.marketinfo import BaseMarketInfo
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.models.order import Order
from lambdatrader.models.orderrequest import OrderRequest
from lambdatrader.models.ordertype import OrderType
from lambdatrader.utilities.utils import pair_from


class BacktestingAccount(BaseAccount):

    def __init__(self, market_info: BaseMarketInfo, balances=None):
        self.market_info = market_info


        self.__balances = defaultdict(int)

        if balances == None:
            balances = {'BTC': 100.0}

        for currency, balance in balances.items():
            self.__balances[currency] = balance

        self.__orders = []

    def get_exchange(self):
        return ExchangeEnum.BACKTESTING

    def get_taker_fee(self, amount):
        return amount * BACKTESTING_TAKER_FEE

    def get_maker_fee(self, amount):
        return amount * BACKTESTING_MAKER_FEE

    def get_balance(self, currency):
        return self.__balances[currency]

    def get_balances(self):
        return dict(self.__balances)

    def get_estimated_balance(self):
        estimated_balance = 0
        for currency, balance in self.__balances.items():
            if currency == 'BTC':
                estimated_balance += balance
            else:
                try:
                    candlestick = self.market_info.get_pair_latest_candlestick(
                                                                pair_from('BTC', currency))
                    estimated_balance += balance * candlestick.close
                except KeyError as e:
                    print('KeyError: ', currency, e)
        for order in self.__orders:
            if not order.get_is_filled():
                if order.get_type() == OrderType.SELL:
                    try:
                        candlestick = self.market_info.get_pair_latest_candlestick(
                            pair_from('BTC', order.get_currency())
                        )
                        estimated_balance += order.get_amount() * candlestick.close
                    except KeyError as e:
                        print('KeyError: ', order.get_currency(), e)
                elif order.get_type() == OrderType.BUY:
                    estimated_balance += order.get_amount() * order.get_price()
        return estimated_balance

    def get_order(self, order_number):
        return self.get_open_orders()[order_number]

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

    def get_open_orders(self):
        open_orders = {}
        for order in self.__orders:
            if not order.get_is_filled():
                open_orders[order.get_order_number()] = order
        return open_orders

    def new_order(self, order_request: OrderRequest, fill_or_kill=False):
        currency = order_request.get_currency()
        order_type = order_request.get_type()
        price = order_request.get_price()
        amount = order_request.get_amount()

        market_date = self.market_info.get_market_date()

        order = Order(currency=currency, _type=order_type,
                      price=price, amount=amount, date=market_date, is_filled=False)

        if order_type == OrderType.SELL:
            if fill_or_kill:
                self.__instant_sell(currency=currency, price=price, amount=amount)
                order.fill()
                return order
            else:
                self.__check_sell_balance_enough(currency, amount)
                self.__balances[currency] -= amount
                self.__orders.append(order)
                return order
        elif order_type == OrderType.BUY:
            if fill_or_kill:
                self.__instant_buy(currency=currency, price=price, amount=amount)
                order.fill()
                return order
            else:
                self.__check_buy_balance_enough(price=price, amount=amount)
                self.__balances['BTC'] -= amount * price
                self.__orders.append(order)
                return order

    def cancel_order(self, order_number):
        order_ind = None
        for i, order in enumerate(self.__orders):
            if order.get_order_number() == order_number:
                order_ind = i
                break
        if order_ind is not None:
            self.__reverse_order_effect(order=self.__orders[order_ind])
            del self.__orders[order_ind]

    def execute_orders(self):
        for order in self.__orders:
            if not order.get_is_filled():
                if self.__order_satisfied(order=order):
                    self.__fill_order(order=order)
                    self.__remove_filled_orders()

    def __order_satisfied(self, order: Order):
        try:
            candlestick = self.market_info.get_pair_latest_candlestick(
                pair_from('BTC', order.get_currency())
            )
            if order.get_type() == OrderType.SELL:
                return candlestick.high >= order.get_price()
            elif order.get_type() == OrderType.BUY:
                return candlestick.low <= order.get_price()
        except KeyError as e:
            print('KeyError: ', order.get_currency(), e)
            return False

    def __fill_order(self, order: Order):
        if order.get_type() == OrderType.SELL:
            btc_value = order.get_amount() * order.get_price()
            self.__balances['BTC'] += btc_value - self.get_maker_fee(amount=btc_value)
        elif order.get_type() == OrderType.BUY:
            balance_addition = order.get_amount() - self.get_maker_fee(order.get_amount())
            self.__balances[order.get_currency()] += balance_addition
        order.fill()

    def __reverse_order_effect(self, order: Order):
        if order.get_type() == OrderType.SELL:
            self.__balances[order.get_currency()] += order.get_amount()
        elif order.get_type() == OrderType.BUY:
            self.__balances['BTC'] += order.get_amount() * order.get_price()

    def __instant_sell(self, currency, price, amount):
        self.__check_instant_sell_valid(currency=currency, price=price, amount=amount)
        self.__balances[currency] -= amount
        self.__balances['BTC'] += amount * price - self.get_taker_fee(amount=amount * price)

    def __check_instant_sell_valid(self, currency, price, amount):
        self.__check_sell_balance_enough(currency=currency, amount=amount)
        self.__check_sell_price_valid(currency=currency, price=price)

    def __check_sell_balance_enough(self, currency, amount):
        if self.__balances[currency] < amount:
            raise NotEnoughBalance(str(amount))

    def __check_sell_price_valid(self, currency, price):
        try:
            if price > self.market_info.get_pair_ticker(pair_from('BTC', currency)).highest_bid:
                raise UnableToFillImmediately
        except KeyError as e:
            print('KeyError: ', currency, e)
            raise UnableToFillImmediately

    def __instant_buy(self, currency, price, amount):
        self.__check_instant_buy_valid(currency=currency, price=price, amount=amount)
        self.__balances[currency] += amount - self.get_taker_fee(amount=amount)
        self.__balances['BTC'] -= amount * price

    def __check_instant_buy_valid(self, currency, price, amount):
        self.__check_buy_balance_enough(price=price, amount=amount)
        self.__check_buy_price_valid(currency=currency, price=price)

    def __check_buy_balance_enough(self, price, amount):
        if self.__balances['BTC'] < price * amount:
            raise NotEnoughBalance(str(amount))

    def __check_buy_price_valid(self, currency, price):
        try:
            if price < self.market_info.get_pair_ticker(pair_from('BTC', currency)).lowest_ask:
                raise UnableToFillImmediately
        except KeyError as e:
            print('KeyError: ', currency, e)
            raise UnableToFillImmediately

    def __remove_filled_orders(self):
        self.__orders = list(filter(lambda order: not order.get_is_filled(), self.__orders))

    def __repr__(self):
        return 'Account(balances={}, orders={})'.format(self.__balances, self.__orders)

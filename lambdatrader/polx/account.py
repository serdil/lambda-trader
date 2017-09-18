from datetime import datetime

from poloniex import PoloniexError

from lambdatrader.account.account import BaseAccount
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.models.order import Order
from lambdatrader.polx.polxclient import polo
from lambdatrader.polx.utils import APICallExecutor
from lambdatrader.utils import pair_second, pair_from
from lambdatrader.models.enums.exchange import ExchangeEnum
from lambdatrader.models.ordertype import OrderType


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
        open_orders = self.get_open_orders()
        open_buy_orders = {}
        for order_number, order in open_orders.items():
            if order.get_type() == OrderType.BUY:
                open_buy_orders[order_number] = order
        return open_buy_orders

    def get_open_sell_orders(self):
        open_orders = self.get_open_orders()
        open_sell_orders = {}
        for order_number, order in open_orders.items():
            if order.get_type() == OrderType.SELL:
                open_sell_orders[order_number] = order
        return open_sell_orders

    @staticmethod
    def __api_call(call):
        return APICallExecutor.get_instance().call(call)


class UnableToFillException(Exception):
    pass
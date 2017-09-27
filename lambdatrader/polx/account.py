from datetime import datetime

from poloniex import PoloniexError

from lambdatrader.config import POLONIEX_TAKER_FEE, POLONIEX_MAKER_FEE
from lambdatrader.account.account import BaseAccount, UnableToFillImmediately
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.models.order import Order
from lambdatrader.polx.polxclient import polo
from lambdatrader.polx.utils import APICallExecutor
from lambdatrader.utils import pair_second, pair_from, get_now_timestamp
from lambdatrader.models.enums.exchange import ExchangeEnum
from lambdatrader.models.ordertype import OrderType


class PolxAccount(BaseAccount):

    def __init__(self):
        self.logger = get_logger_with_all_handlers(__name__)

    def get_exchange(self):
        return ExchangeEnum.POLONIEX

    def get_taker_fee(self, amount):
        return amount * POLONIEX_TAKER_FEE

    def get_maker_fee(self, amount):
        return amount * POLONIEX_MAKER_FEE

    def get_balance(self, currency):
        return self.get_balances()[currency]

    def get_balances(self):
        self.logger.debug('get_balances')
        balances = {}
        balances_response = self.__api_call(call=lambda: polo.returnBalances())
        for currency, balance in balances_response.items():
            balances[currency] = float(balance)
        return balances

    def get_estimated_balance(self):
        self.logger.debug('get_estimated_balance')
        complete_balances = self.__api_call(call=lambda: polo.returnCompleteBalances())

        estimated_balance = 0.0
        for info in complete_balances.values():
            estimated_balance += float(info['btcValue'])

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
        order_number = value['orderNumber']
        amount = float(value['amount'])
        price = float(value['rate'])
        return Order(currency=currency, _type=_type, price=price,
                     amount=amount, date=date, order_number=order_number)

    def new_order(self, order_request, fill_or_kill=False):
        self.logger.info('new_order_request:%s', str(order_request))
        currency = order_request.get_currency()
        order_type = order_request.get_type()
        price = order_request.get_price()
        amount = order_request.get_amount()

        polx_market_date = get_now_timestamp()

        order = Order(currency=currency, _type=order_type,
                      price=price, amount=amount, date=polx_market_date, is_filled=bool(fill_or_kill))
        try:
            order_result = self.__polo_put(order_request=order_request, fill_or_kill=fill_or_kill)
            order.set_order_number(order_result['orderNumber'])
            self.logger.info('order_put:%s', order.get_order_number())
            return order
        except PoloniexError as e:
            if str(e) == 'Unable to fill order completely.':
                raise UnableToFillImmediately
            else:
                raise e

    def cancel_order(self, order_number):
        self.logger.info('cancel_order:%d', order_number)
        self.__api_call(call=lambda: polo.cancelOrder(order_number))
        self.logger.info('order_cancelled:%d', order_number)

    def __polo_put(self, order_request, fill_or_kill=False):
        if order_request.get_amount() == -1:
            amount = self.get_balance(order_request.get_currency())
        else:
            amount = order_request.get_amount()

        if order_request.get_type() == OrderType.BUY:
            buy_result = self.__api_call(
                lambda: polo.buy(currencyPair=pair_from('BTC', order_request.get_currency()),
                                 rate=order_request.get_price(),
                                 amount=amount,
                                 orderType='fillOrKill' if fill_or_kill else False)
            )
            return buy_result
        elif order_request.get_type() == OrderType.SELL:
            sell_result = self.__api_call(
                lambda: polo.sell(currencyPair=pair_from('BTC', order_request.get_currency()),
                                  rate=order_request.get_price(),
                                  amount=amount,
                                  orderType='fillOrKill' if fill_or_kill else False)
            )
            return sell_result

    @staticmethod
    def __api_call(call):
        return APICallExecutor.get_instance().call(call)


class UnableToFillException(Exception):
    pass

from uuid import uuid1

from lambdatrader.models.ordertype import OrderType


class OrderRequest:
    def __init__(self, currency, _type: OrderType, price, amount, date, order_request_number=None):
        self.__currency = currency
        self.__type = _type
        self.__price = price
        self.__amount = amount
        self.__date = date

        if order_request_number is not None:
            self.order_request_number = order_request_number
        else:
            self.order_request_number = uuid1()

    def __repr__(self):
        return 'OrderRequest(' + str(self.__currency) + ' ' + str(self.__type) + \
               ' price=' + str(self.__price) + ' amount=' + str(self.__amount) +\
               ' date=' + str(self.__date) + \
               ' order_request_number=' + str(self.order_request_number) + ')'

    def get_order_number(self):
        return self.order_request_number

    def get_currency(self):
        return self.__currency

    def get_type(self):
        return self.__type

    def get_price(self):
        return self.__price

    def get_amount(self):
        return self.__amount

    def get_date(self):
        return self.__date

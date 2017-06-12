from enum import Enum
from uuid import uuid1


class OrderType(Enum):
    BUY = 1
    SELL = 2


class Order:
    def __init__(self, currency, type: OrderType, price, amount, timestamp, is_filled=False, order_number=uuid1()):
        self.__currency = currency
        self.__type = type
        self.__price = price
        self.__amount = amount
        self.__timestamp = timestamp
        self.__is_filled = is_filled
        self.__order_number = order_number

    def fill(self):
        self.__is_filled = True

    def __repr__(self):
        return 'Order(' + str(self.__currency) + ' ' + str(self.__type) + ' price=' + str(self.__price) + \
               ' amount=' + str(self.__amount) + ' timestamp=' + str(self.__timestamp) + ' is_filled=' + str(self.__is_filled) + ')'

    def get_order_number(self):
        return self.__order_number

    def get_currency(self):
        return self.__currency

    def get_type(self):
        return self.__type

    def get_price(self):
        return self.__price

    def get_amount(self):
        return self.__amount

    def get_timestamp(self):
        return self.__timestamp

    def get_is_filled(self):
        return self.__is_filled
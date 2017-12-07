from uuid import uuid1

from lambdatrader.models.ordertype import OrderType


class Order:
    def __init__(self, currency, _type: OrderType, price, amount, date, is_filled=False,
                 order_number=None):
        self.__currency = currency
        self.__type = _type
        self.__price = price
        self.__amount = amount
        self.__date = date
        self.__is_filled = is_filled

        if order_number is not None:
            self.__order_number = order_number
        else:
            self.__order_number = uuid1()

    def fill(self):
        self.__is_filled = True

    def __repr__(self):
        return 'Order(' + str(self.__currency) + ' ' + str(self.__type) + \
               ' price=' + str(self.__price) + ' amount=' + str(self.__amount) +\
               ' date=' + str(self.__date) + ' is_filled=' + str(self.__is_filled) + \
               ' order_number=' + str(self.__order_number) + ')'

    def set_order_number(self, order_number):
        self.__order_number = order_number

    @property
    def order_number(self):
        return self.get_order_number()

    @property
    def currency(self):
        return self.currency

    @property
    def type(self):
        return self.type

    @property
    def price(self):
        return self.price

    @property
    def amount(self):
        return self.amount

    @property
    def date(self):
        return self.date

    @property
    def is_filled(self):
        return self.is_filled

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

    def get_date(self):
        return self.__date

    def get_is_filled(self):
        return self.__is_filled

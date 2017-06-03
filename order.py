from enum import Enum
from uuid import uuid1

from currency import Currency


class OrderType(Enum):
    BUY = 1
    SELL = 2


class Order:
    def __init__(self, currency: Currency, type: OrderType, price, amount, timestamp, is_filled=False):
        self.currency = currency
        self.type = type
        self.price = price
        self.amount = amount
        self.timestamp = timestamp
        self.is_filled = is_filled
        self.id = uuid1()

    def fill(self):
        self.is_filled = True

    def __repr__(self):
        return 'Order(' + str(self.currency) + ' ' + str(self.type) + ' price=' + str(self.price) + \
               ' amount=' + str(self.amount) + ' timestamp=' + str(self.timestamp) + ' is_filled=' + str(self.is_filled) + ')'
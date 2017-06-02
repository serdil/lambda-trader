from enum import Enum

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

    def fill(self):
        self.is_filled = True
class Order:
    def __init__(self, currency, type, price, value, timestamp, is_filled=False):
        self.currency = currency
        self.type = type
        self.price = price
        self.value = value
        self.timestamp = timestamp
        self.is_filled = is_filled

    def fill(self):
        self.is_filled = True
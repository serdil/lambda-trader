class Trade:
    def __init__(self, _id, _type, pair, rate, amount, fee, total, date):
        self.id = _id
        self.type = _type
        self.pair = pair
        self.rate = rate
        self.amount = amount
        self.fee = fee
        self.total = total
        self.date = date

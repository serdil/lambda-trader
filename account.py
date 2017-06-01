
class Account:
    def __init__(self, balances={'BTC': 100}, orders=[]):
        self.balances = {}
        for currency, balance in balances:
            self.balances[currency] = balance
        self.orders = []
        for order in orders:
            self.orders.append(order)


class BaseAccount:

    def get_exchange(self):
        raise NotImplementedError

    def get_estimated_balance(self, market_info):
        raise NotImplementedError

    def get_open_sell_orders(self):
        raise NotImplementedError

    def get_open_buy_orders(self):
        raise NotImplementedError

    def get_open_orders(self):
        raise NotImplementedError

    def get_order(self, order_number):
        raise NotImplementedError

    def cancel_order(self, order_number):
        raise NotImplementedError

    def sell(self, currency, price, amount):
        raise NotImplementedError

    def get_taker_fee(self, amount):
        raise NotImplementedError

    def get_maker_fee(self, amount):
        raise NotImplementedError

    def get_balance(self, currency):
        raise NotImplementedError

    def get_balances(self):
        raise NotImplementedError

    def buy(self, currency, price, amount):
        raise NotImplementedError

    def new_order(self, order_request, fill_or_kill=False):
        raise NotImplementedError

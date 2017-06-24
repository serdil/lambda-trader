from models.candlestick import Candlestick


class CandlestickStore:

    def __init__(self):
        pass

    def add_candlestick(self, pair, candlestick: Candlestick):
        pass

    def get_candlestick(self, pair, ind=0):
        pass

    def get_pair_oldest_date(self, pair):
        pass

    def get_pair_newest_date(self, pair):
        pass

    def __create_pair_table(self, pair):
        pass

from datetime import datetime

from models.order import Order, OrderType
from utils import pair_from, pair_second
from .strategy import BaseStrategy


class InternalTrade:
    def __init__(self, currency, amount, rate, target_rate):
        self.currency = currency
        self.amount = amount
        self.rate = rate
        self.target_rate = target_rate


class BacktestStrategy(BaseStrategy):

    DELTA = 0.0001

    PERIOD = 300
    ORDER_TIMEOUT = 1 * 24 * 3600

    NUM_CHUNKS = 10
    HIGH_VOLUME_LIMIT = 20
    MIN_CHUNK_SIZE = 0.00011
    MIN_NUM_HIGH_VOLUME_PAIRS = 1
    BUY_PROFIT_FACTOR = 1.03

    RETRACEMENT_RATIO = 0.1

    def __init__(self):
        super().__init__()
        self.__trades = {}

    def act(self, account, market_info):
        self.__declare_successfuly_closed_trades(market_info, account)
        self.cancel_old_orders(account, market_info)

        high_volume_pairs = sorted(self.get_high_volume_pairs(market_info),
                                   key=lambda pair: -market_info.get_pair_last_24h_btc_volume(pair))

        open_pairs = self.get_pairs_with_open_orders(account)

        estimated_balance = account.get_estimated_balance(market_info)

        self.declare_balance(market_info.get_market_time, estimated_balance)

        if len(high_volume_pairs) >= self.MIN_NUM_HIGH_VOLUME_PAIRS:
            for pair in high_volume_pairs:
                if pair not in open_pairs:
                    self.act_on_pair(account, market_info, pair, estimated_balance)

    def act_on_pair(self, account, market_info, pair, estimated_balance):
        chunk_size = estimated_balance / self.NUM_CHUNKS

        if chunk_size >= self.MIN_CHUNK_SIZE:

            latest_ticker = market_info.get_pair_ticker(pair)
            price = latest_ticker.lowest_ask
            market_date = market_info.get_market_time()

            if account.get_balance('BTC') >= chunk_size * (1.0 + self.DELTA):
                target_price = price * self.BUY_PROFIT_FACTOR
                day_high_price = latest_ticker.high24h

                price_is_lower_than_day_high = target_price < day_high_price

                if price_is_lower_than_day_high:
                    current_retracement_ratio = (target_price - price) / (day_high_price - price)
                    retracement_ratio_satisfied = \
                        current_retracement_ratio <= self.RETRACEMENT_RATIO

                    if retracement_ratio_satisfied:
                        print(datetime.fromtimestamp(market_info.get_market_time()))

                        currency = pair_second(pair)

                        account.buy(currency, price, chunk_size / price, market_info)

                        bought_amount = account.get_balance(pair_second(pair))

                        sell_order = Order(currency, OrderType.SELL, target_price,
                                           bought_amount, market_date)

                        order_number = sell_order.get_order_number()

                        self.declare_trade_start(market_date, order_number)

                        self.__trades[order_number] = \
                            InternalTrade(currency, bought_amount, price, target_price)

                        current_balance = estimated_balance
                        max_drawback, avg_drawback = account.max_avg_drawback()
                        account.new_order(sell_order)

                        print('BUY', pair)
                        print('balance', current_balance)
                        print('max-avg drawback', max_drawback, avg_drawback)
                        print('open orders:', len(list(account.get_open_orders())))

    def cancel_old_orders(self, account, market_info):
        order_numbers_to_cancel = []

        for order in account.get_open_orders():
            if market_info.get_market_time() - order.get_timestamp() >= self.ORDER_TIMEOUT:
                order_numbers_to_cancel.append(order.get_order_number())

        for order_number in order_numbers_to_cancel:

            print('cancelling')

            order = account.get_order(order_number)
            account.cancel_order(order_number)
            price = market_info.get_pair_ticker(pair_from('BTC', order.get_currency())).highest_bid
            account.sell(order.get_currency(), price, order.get_amount(), market_info)

            trade = self.__trades[order_number]
            profit_amount = trade.amount * price - trade.amount * trade.rate  # should be < 0

            self.declare_trade_end(market_info.get_market_time(), order_number, profit_amount)

    @staticmethod
    def get_pairs_with_open_orders(account):
        return set([pair_from('BTC', order.get_currency()) for order in account.get_open_orders()])

    def get_high_volume_pairs(self, market_info):
        return list(
            filter(lambda p: market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                   market_info.pairs())
        )

    def __declare_successfuly_closed_trades(self, market_info, account):
        open_orders_set = set(account.get_open_orders)

        for trade_number, trade in self.__trades.items():
            if trade_number not in open_orders_set:  # trade hit TP
                close_date = market_info.get_market_time()
                profit_amount = trade.amount * trade.target_rate - trade.amount * trade.rate
                self.declare_trade_end(close_date, trade_number, profit_amount)

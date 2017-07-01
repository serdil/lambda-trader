from datetime import datetime

from models.order import Order, OrderType
from utils import pair_from, pair_second
from .strategy import BaseStrategy


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

    def act(self, account, market_info):
        self.cancel_old_orders(account, market_info)

        high_volume_pairs = sorted(self.get_high_volume_pairs(market_info),
                                   key=lambda pair: -market_info.get_pair_last_24h_btc_volume(pair))

        open_pairs = self.get_pairs_with_open_orders(account)

        estimated_balance = account.get_estimated_balance(market_info)

        if len(high_volume_pairs) >= self.MIN_NUM_HIGH_VOLUME_PAIRS:
            for pair in high_volume_pairs:
                if pair not in open_pairs:
                    self.act_on_pair(account, market_info, pair, estimated_balance)

    def act_on_pair(self, account, market_info, pair, estimated_balance):
        chunk_size = estimated_balance / self.NUM_CHUNKS

        if chunk_size >= self.MIN_CHUNK_SIZE:

            latest_ticker = market_info.get_pair_ticker(pair)
            price = latest_ticker.lowest_ask
            timestamp = market_info.get_market_time()

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

                        account.buy(pair_second(pair), price, chunk_size / price, market_info)

                        sell_order = Order(pair_second(pair), OrderType.SELL, target_price,
                                           account.get_balance(pair_second(pair)), timestamp)

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

    def get_pairs_with_open_orders(self, account):
        return set([pair_from('BTC', order.get_currency()) for order in account.get_open_orders()])

    def get_high_volume_pairs(self, market_info):
        return list(
            filter(lambda p: market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                   market_info.pairs())
        )
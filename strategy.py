from datetime import datetime

from currency import Currency
from currencypair import CurrencyPair
from marketinfo import MarketInfo
from order import Order, OrderType


class Strategy:

    DELTA = 0.0001

    PERIOD = 300
    ORDER_TIMEOUT = 1 * 24 * 3600

    NUM_CHUNKS = 25
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

        if len(high_volume_pairs) >= self.MIN_NUM_HIGH_VOLUME_PAIRS:
            for pair in high_volume_pairs:
                if pair not in open_pairs:
                    self.act_on_pair(account, market_info, pair)

    def act_on_pair(self, account, market_info: MarketInfo, pair: CurrencyPair):
        chunk_size = account.get_estimated_balance(market_info) / self.NUM_CHUNKS
        if chunk_size >= self.MIN_CHUNK_SIZE:
            latest_candlestick = market_info.get_pair_latest_candlestick(pair)
            price = latest_candlestick.close
            timestamp = latest_candlestick.timestamp
            if account.get_balance(Currency.BTC) >= chunk_size * (1.0 + self.DELTA):
                target_price = price * self.BUY_PROFIT_FACTOR
                day_high_price = self.get_24h_high_price(market_info, pair)

                if target_price < day_high_price and (target_price - price) / (day_high_price - price) <= self.RETRACEMENT_RATIO:

                    print(datetime.fromtimestamp(market_info.get_market_time()))

                    account.buy(pair.second, price, chunk_size / price, market_info)

                    sell_order = Order(pair.second, OrderType.SELL, target_price,
                                       account.get_balance(pair.second), timestamp)

                    current_balance = account.get_estimated_balance(market_info)
                    max_drawback, avg_drawback = account.max_avg_drawback()
                    account.new_order(sell_order)

                    print('balance', current_balance)
                    print('max-avg drawback', max_drawback, avg_drawback)
                    print('open orders:', len(list(account.get_open_orders())))


    def cancel_old_orders(self, account, market_info):
        order_ids_to_cancel = []
        for order in account.get_open_orders():
            if market_info.get_market_time() - order.timestamp >= self.ORDER_TIMEOUT:
                order_ids_to_cancel.append(order.id)
        for id in order_ids_to_cancel:
            print('cancelling')
            order = account.get_order(id)
            account.cancel_order(id)
            price = market_info.get_pair_latest_candlestick(
                CurrencyPair(Currency.BTC, order.currency)).close
            account.sell(order.currency, price, order.amount)

    def get_24h_high_price(self, market_info, pair):
        return market_info.get_pair_last_24h_high(pair)

    def get_pairs_with_open_orders(self, account):
        return set([CurrencyPair(Currency.BTC, order.currency) for order in account.get_open_orders()])

    def get_high_volume_pairs(self, market_info):
        return [kv[0] for kv in filter(lambda kv: market_info.get_pair_last_24h_btc_volume(kv[0]) >= self.HIGH_VOLUME_LIMIT,
                      market_info.pairs())]
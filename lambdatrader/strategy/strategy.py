from datetime import datetime
from enum import Enum
from threading import Thread
from time import sleep

from loghandlers import get_logger_with_all_handlers
from models.order import Order, OrderType

from polx.polxdriver import PolxAccount, UnableToFillException
from utils import pair_second, pair_from, get_now_timestamp


class Direction(Enum):
    UP = 1
    DOWN = 2


class VolumeStrategy:

    DELTA = 0.0001

    ORDER_TIMEOUT = 6 * 3600  # in seconds

    NUM_CHUNKS = 10
    HIGH_VOLUME_LIMIT = 0
    MIN_CHUNK_SIZE = 0.00011
    MIN_NUM_HIGH_VOLUME_PAIRS = 0

    RECENT_VOLUME_PERIOD = 6  # in number of candlesticks
    LOOKBACK_VOLUME_PERIOD = 6 * 6
    RECENT_VOLUME_THRESHOLD_PERCENT = 100
    VOLUME_DIRECTION = Direction.DOWN

    LOOKBACK_PRICE_PERIOD = 6  # in number of candlesticks
    PRICE_INCREASE_THRESHOLD_PERCENT = -6
    PRICE_DIRECTION = Direction.DOWN

    PROFIT_TARGET_PERCENT = 3

    def __init__(self, mult=1):
        self.ORDER_TIMEOUT *= mult  # in seconds
        self.RECENT_VOLUME_PERIOD *= mult
        self.LOOKBACK_VOLUME_PERIOD *= mult
        self.RECENT_VOLUME_THRESHOLD_PERCENT *= mult
        self.LOOKBACK_PRICE_PERIOD *= mult
        self.PRICE_INCREASE_THRESHOLD_PERCENT *= mult
        self.PROFIT_TARGET_PERCENT *= mult

        self.ORDER_TIMEOUT = int(self.ORDER_TIMEOUT)  # in seconds
        self.RECENT_VOLUME_PERIOD = int(self.RECENT_VOLUME_PERIOD)
        self.LOOKBACK_VOLUME_PERIOD = int(self.LOOKBACK_VOLUME_PERIOD)
        self.RECENT_VOLUME_THRESHOLD_PERCENT = int(self.RECENT_VOLUME_THRESHOLD_PERCENT)
        self.LOOKBACK_PRICE_PERIOD = int(self.LOOKBACK_PRICE_PERIOD)
        self.PRICE_INCREASE_THRESHOLD_PERCENT = int(self.PRICE_INCREASE_THRESHOLD_PERCENT)
        self.PROFIT_TARGET_PERCENT = int(self.PROFIT_TARGET_PERCENT)

        self.__jibun_ga_aketa = []

    def act(self, account, market_info):
        self.cancel_old_orders(account, market_info)

        high_volume_pairs = sorted(self.get_high_volume_pairs(market_info),
                                   key=lambda pair: -market_info.get_pair_last_24h_btc_volume(pair))

        open_pairs = self.get_pairs_with_open_orders(account)

        estimated_balance = account.get_estimated_balance(market_info)

        if len(high_volume_pairs) >= self.MIN_NUM_HIGH_VOLUME_PAIRS:
            for pair in high_volume_pairs:
                if pair not in open_pairs:
                    try:
                        self.act_on_pair(account, market_info, pair, estimated_balance)
                    except KeyError:
                        pass

    def act_on_pair(self, account, market_info, pair, estimated_balance):
        chunk_size = estimated_balance / self.NUM_CHUNKS

        if chunk_size >= self.MIN_CHUNK_SIZE:

            if account.get_balance('BTC') >= chunk_size * (1.0 + self.DELTA):

                old_candlestick = market_info.get_pair_candlestick(pair, self.LOOKBACK_PRICE_PERIOD)
                latest_ticker = market_info.get_pair_ticker(pair)

                old_price = old_candlestick.close
                current_price = latest_ticker.lowest_ask
                timestamp = market_info.get_market_time()

                recent_volume = self.calc_pair_recent_volume(market_info, pair,
                                                             self.RECENT_VOLUME_PERIOD)

                if recent_volume == 0:
                    return

                lookback_volume = self.calc_pair_recent_volume(market_info, pair,
                                                               self.LOOKBACK_VOLUME_PERIOD)

                target_price = current_price * ((100 + self.PROFIT_TARGET_PERCENT) / 100)

                recent_volume_percent = (recent_volume / lookback_volume) * 100
                price_increase_percent = (current_price - old_price) / current_price * 100

                if self.VOLUME_DIRECTION is Direction.UP:
                    volume_cond_satisfied = recent_volume_percent >= \
                                            self.RECENT_VOLUME_THRESHOLD_PERCENT
                elif self.VOLUME_DIRECTION is Direction.DOWN:
                    volume_cond_satisfied = recent_volume_percent <= \
                                            self.RECENT_VOLUME_THRESHOLD_PERCENT
                else:
                    volume_cond_satisfied = False

                if self.PRICE_DIRECTION is Direction.UP:
                    price_cond_satisfied = price_increase_percent >= \
                                           self.PRICE_INCREASE_THRESHOLD_PERCENT
                elif self.PRICE_DIRECTION is Direction.DOWN:
                    price_cond_satisfied = price_increase_percent <= \
                                           self.PRICE_INCREASE_THRESHOLD_PERCENT
                else:
                    price_cond_satisfied = False

                if volume_cond_satisfied and price_cond_satisfied:
                    print(datetime.fromtimestamp(market_info.get_market_time()))
                    account.buy(pair_second(pair), current_price, chunk_size / current_price, market_info)

                    sell_order = Order(pair_second(pair), OrderType.SELL, target_price,
                                       account.get_balance(pair_second(pair)), timestamp)

                    self.__jibun_ga_aketa.append(sell_order.get_order_number())

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
                if order.get_order_number() in self.__jibun_ga_aketa:
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
        return list(filter(lambda p: market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                      market_info.pairs()))

    def calc_pair_recent_volume(self, market_info, pair, lookback):
        recent_volume = 0
        for i in range(lookback):
            recent_volume += market_info.get_pair_candlestick(pair, i).base_volume
        return recent_volume


class Strategy:

    DELTA = 0.0001

    ORDER_TIMEOUT = 1 * 24 * 3600

    NUM_CHUNKS = 10
    HIGH_VOLUME_LIMIT = 20
    MIN_CHUNK_SIZE = 0.00011
    MIN_NUM_HIGH_VOLUME_PAIRS = 1
    BUY_PROFIT_FACTOR = 1.03

    RETRACEMENT_RATIO = 0.10

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

                if target_price < day_high_price and (target_price - price) / (day_high_price - price) <= self.RETRACEMENT_RATIO:

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
        return list(filter(lambda p: market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                      market_info.pairs()))


class PolxStrategy:

    DELTA = 0.0001

    ORDER_TIMEOUT = 1 * 24 * 3600

    NUM_CHUNKS = 10
    HIGH_VOLUME_LIMIT = 20
    MIN_CHUNK_SIZE = 0.00011
    MIN_NUM_HIGH_VOLUME_PAIRS = 1
    BUY_PROFIT_FACTOR = 1.03

    RETRACEMENT_RATIO = 0.1

    def __init__(self, market_info, account: PolxAccount):
        self.market_info = market_info
        self.account = account

        self.__balances = {}
        self.__balances_last_updated = 0

        self.__estimated_balance = 0
        self.__estimated_balance_last_updated = 0

        self.__balance_series = []

        self.logger = get_logger_with_all_handlers(__name__)

        self.__update_estimated_balance()
        self.__start_heartbeat_thread()

    def __start_heartbeat_thread(self):
        self.logger.debug('starting_heartbeat_thread')
        t = Thread(target=self.heartbeat_thread)
        t.start()

    def act(self):
        self.logger.debug('acting')

        self.__cancel_old_orders()

        high_volume_pairs = self.__get_high_volume_pairs()

        self.logger.debug('num_high_volume_pairs: %d', len(high_volume_pairs))

        open_pairs = self.__get_pairs_with_open_orders()
        self.__num_open_pairs = len(open_pairs)

        self.logger.debug('open_pairs: %s', self.join(open_pairs))

        self.__update_balances()
        self.__update_estimated_balance()

        self.__sample_balance()

        estimated_balance = self.__get_estimated_balance()
        chunk_size = estimated_balance / self.NUM_CHUNKS

        self.logger.debug('estimated_balance: %f', estimated_balance)
        self.logger.debug('chunk_size: %f', chunk_size)

        if len(high_volume_pairs) >= self.MIN_NUM_HIGH_VOLUME_PAIRS:
            for pair in high_volume_pairs:
                if pair not in open_pairs:
                    self.act_on_pair(pair)

    def act_on_pair(self, pair):
        self.logger.debug('acting_on: %s', pair)
        self.__update_balances_if_outdated()
        chunk_size = self.__get_estimated_balance() / self.NUM_CHUNKS
        self.logger.debug('chunk_size: %f', chunk_size)
        if chunk_size >= self.MIN_CHUNK_SIZE:
            self.logger.debug('chunk size enough')

            latest_ticker = self.market_info.get_pair_ticker(pair)
            self.logger.debug('latest_ticker: %s', str(latest_ticker))

            price = latest_ticker.lowest_ask
            self.logger.debug('price: %f', price)

            timestamp = self.market_info.get_market_time()
            self.logger.debug('timestamp: %d', timestamp)

            if self.__get_balance('BTC') >= chunk_size * (1.0 + self.DELTA):
                self.logger.debug('btc balance is enough')

                target_price = price * self.BUY_PROFIT_FACTOR
                day_high_price = latest_ticker.high24h

                self.logger.debug('target_price: %f', target_price)
                self.logger.debug('day_high_price: %f', day_high_price)

                if day_high_price > price:
                    target_day_high_ratio = (target_price - price) / (day_high_price - price)
                else:
                    target_day_high_ratio = float('inf')

                self.logger.debug('target_day_high_ratio: %f', target_day_high_ratio)

                if target_price < day_high_price and target_day_high_ratio <= self.RETRACEMENT_RATIO:
                    self.logger.info('retracement ratio satisfied, attempting trade')

                    try:
                        buy_order = Order(pair_second(pair), OrderType.BUY, price,
                                          chunk_size / price, timestamp)
                        self.logger.info('buy_order: %s', str(buy_order))

                        self.account.new_order(buy_order, fill_or_kill=True)
                        sell_order = Order(pair_second(pair), OrderType.SELL, target_price, -1, timestamp)
                        self.account.new_order(sell_order)
                        self.__update_balances()

                        current_balance = self.__get_estimated_balance()
                        max_drawback, avg_drawback = self.__max_avg_drawback()

                        self.logger.info('trade successful')
                        self.logger.info('balance: %f', current_balance)
                        self.logger.info('max_avg_drawback: %f %f', max_drawback, avg_drawback)

                        self.__num_open_pairs += 1
                        self.logger.info('num_open_orders: %d', self.__num_open_pairs)
                    except UnableToFillException:
                        self.logger.warning('unable to fill order immediately')

    def __cancel_old_orders(self):
        self.logger.debug('cancel_old_orders')

        orders_to_cancel = []

        for order in self.account.get_open_orders().values():
            if self.market_info.get_market_time() - order.get_timestamp() >= self.ORDER_TIMEOUT:
                orders_to_cancel.append(order)

        self.logger.debug('orders_to_cancel: %s', self.join(orders_to_cancel))

        for order in orders_to_cancel:
            self.logger.info('cancelling_order: %s', str(order))
            self.account.cancel_order(order.get_order_number())

            price = self.market_info.get_pair_ticker(pair_from('BTC',order.get_currency())).lowest_ask
            self.logger.info('cancelling_price: %f', price)

            sell_order = self.make_order(order.get_currency(), price,
                                         -1, OrderType.SELL,
                                         self.market_info.get_market_time())
            self.account.new_order(sell_order)
            self.logger.info('sell_order_put')

            self.__update_balances()

    def __get_pairs_with_open_orders(self):
        self.logger.debug('get_pairs_with_open_orders')
        return set([pair_from('BTC', order.get_currency()) for order in self.account.get_open_orders().values()])

    def __get_high_volume_pairs(self):
        self.logger.debug('get_high_volume_pairs')
        return sorted(
            list(
                filter(
                    lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                    self.market_info.pairs()
                )
            ),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair)
        )

    def __get_balance(self, currency):
        return float(self.__balances[currency])

    def __update_balances_if_outdated(self):
        self.logger.debug('update_balances_if_outdated')

        if get_now_timestamp() - self.__balances_last_updated > 5:
            self.__update_balances()

    def __update_balances(self):
        self.logger.debug('updating_balances')

        self.__balances = self.account.get_balances()
        self.__balances_last_updated = get_now_timestamp()

    def __get_estimated_balance(self):
        return self.__estimated_balance

    def __update_estimated_balance_if_outdated(self):
        self.logger.debug('update_estimated_balance_if_outdated')

        if get_now_timestamp() - self.__estimated_balance_last_updated > 5:
            self.__update_estimated_balance()

    def __update_estimated_balance(self):
        self.logger.debug('updating_estimated_balance')

        self.__estimated_balance = self.account.get_estimated_balance()
        self.__estimated_balance_last_updated = get_now_timestamp()

    def __sample_balance(self):
        estimated_balance = self.__get_estimated_balance()
        self.logger.debug('sampling_balance: %f', estimated_balance)

        self.__balance_series.append(estimated_balance)

    def __max_avg_drawback(self):
        total_drawback = 0.0
        num_drawbacks = 0
        max_drawback = 0.0
        max_balance_so_far = 0.0
        for balance in self.__balance_series:
            if balance > max_balance_so_far:
                max_balance_so_far = balance
            if balance < max_balance_so_far:
                num_drawbacks += 1
                current_drawback = (max_balance_so_far - balance) / max_balance_so_far * 100
                total_drawback += current_drawback
                if current_drawback > max_drawback:
                    max_drawback = current_drawback
        return max_drawback, (total_drawback / num_drawbacks if num_drawbacks > 0 else 0.0)

    def heartbeat_thread(self):
        while True:
            self.__log_heartbeat_info_conditionally(log_if_no_open_orders=True)
            sleep(1800)
            self.__log_heartbeat_info_conditionally(log_if_no_open_orders=False)
            sleep(1800)
            self.__log_heartbeat_info_conditionally(log_if_no_open_orders=False)
            sleep(1800)
            self.__log_heartbeat_info_conditionally(log_if_no_open_orders=False)
            sleep(1800)

    def __log_heartbeat_info_conditionally(self, log_if_no_open_orders=False):
        num_open_orders = len(self.__get_pairs_with_open_orders())
        if num_open_orders == 0 and not log_if_no_open_orders:
            return
        estimated_balance = self.__get_estimated_balance()
        self.__log_heartbeat_info(estimated_balance, num_open_orders)

    def __log_heartbeat_info(self, estimated_balance, num_open_orders):
        self.logger.info('HEARTBEAT: estimated_balance: %f num_open_orders: %d',
            self.__get_estimated_balance(), len(self.__get_pairs_with_open_orders()))

    @staticmethod
    def make_order(currency, price, amount, order_type, timestamp):
        return Order(currency=currency, price=price, amount=amount, type=order_type, timestamp=timestamp)

    @staticmethod
    def join(items):
        return ' '.join([str(item) for item in items])


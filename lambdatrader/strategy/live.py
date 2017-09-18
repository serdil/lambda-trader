from threading import Thread
from time import sleep

from poloniex import PoloniexError

from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT, EXECUTOR__NUM_CHUNKS,
    RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT, EXECUTOR__MIN_CHUNK_SIZE,
    RETRACEMENT_STRATEGY__MIN_NUM_HIGH_VOLUME_PAIRS, RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR,
    RETRACEMENT_SIGNALS__RETRACEMENT_RATIO, POLONIEX_TRADING_PAIRS,
)
from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.models.order import Order
from lambdatrader.models.ordertype import OrderType
from lambdatrader.polx.account import PolxAccount, UnableToFillException
from lambdatrader.utils import pair_second, pair_from, get_now_timestamp


# TODO Use signals and executors for these.

class PolxStrategy:

    DELTA = 0.0001

    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT

    NUM_CHUNKS = EXECUTOR__NUM_CHUNKS
    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    MIN_CHUNK_SIZE = EXECUTOR__MIN_CHUNK_SIZE
    MIN_NUM_HIGH_VOLUME_PAIRS = RETRACEMENT_STRATEGY__MIN_NUM_HIGH_VOLUME_PAIRS
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR

    RETRACEMENT_RATIO = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    def __init__(self, market_info, account: PolxAccount):
        self.market_info = market_info
        self.account = account

        self.__balances = {}
        self.__balances_last_updated = 0

        self.__estimated_balance = 0
        self.__estimated_balance_last_updated = 0

        self.__balance_series = []

        self.__last_order_cancellation_time = 0

        self.logger = get_logger_with_all_handlers(__name__)

        self.__update_estimated_balance()
        self.__start_heartbeat_thread()

        self.__allowed_trading_pairs_set = set(POLONIEX_TRADING_PAIRS)

    def __start_heartbeat_thread(self):
        self.logger.debug('starting_heartbeat_thread')
        t = Thread(target=self.heartbeat_thread)
        t.start()

    def act(self):
        self.logger.debug('acting')

        if self.market_info.get_market_date() - self.__last_order_cancellation_time >= 1800:
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
                if pair not in open_pairs and pair in POLONIEX_TRADING_PAIRS:
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

            market_date = self.market_info.get_market_date()
            self.logger.debug('date: %d', market_date)

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

                price_is_lower_than_day_high = target_price < day_high_price
                retracement_ratio_satisfied = target_day_high_ratio <= self.RETRACEMENT_RATIO

                if price_is_lower_than_day_high and retracement_ratio_satisfied:
                    self.logger.info('retracement ratio satisfied, attempting trade')

                    try:
                        buy_order = Order(currency=pair_second(pair),
                                          _type=OrderType.BUY,
                                          price=price,
                                          amount=chunk_size / price,
                                          date=market_date)
                        self.logger.info('buy_order: %s', str(buy_order))

                        self.account.new_order(order=buy_order, fill_or_kill=True)
                        sell_order = Order(currency=pair_second(pair),
                                           _type=OrderType.SELL,
                                           price=target_price,
                                           amount=-1,
                                           date=market_date)
                        self.account.new_order(order=sell_order)
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
            if self.market_info.get_market_date() - order.get_date() >= self.ORDER_TIMEOUT:
                orders_to_cancel.append(order)

        self.logger.debug('orders_to_cancel: %s', self.join(orders_to_cancel))

        for order in orders_to_cancel:
            self.logger.info('cancelling_order: %s', str(order))
            self.account.cancel_order(order.get_order_number())

            price = self.market_info.get_pair_ticker(
                pair_from('BTC', order.get_currency())
            ).lowest_ask
            self.logger.info('cancelling_price: %f', price)

            sell_order = self.make_order(order.get_currency(), price,
                                         -1, OrderType.SELL,
                                         self.market_info.get_market_date())
            self.account.new_order(order=sell_order)
            self.logger.info('sell_order_put')

            self.__update_balances()

        self.__last_order_cancellation_time = self.market_info.get_market_date()

    def __get_pairs_with_open_orders(self):
        self.logger.debug('get_pairs_with_open_orders')
        return set(
            [pair_from('BTC', order.get_currency()) for
             order in self.account.get_open_orders().values()]
        )

    def __get_high_volume_pairs(self):
        self.logger.debug('get_high_volume_pairs')
        return sorted(
            list(
                filter(
                    lambda p:
                    self.market_info.get_pair_last_24h_btc_volume(pair=p) >= self.HIGH_VOLUME_LIMIT,
                    self.market_info.get_active_pairs()
                )
            ),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
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
            try:
                self.__log_heartbeat_info_conditionally(log_if_no_open_orders=True)
                sleep(1800)
                self.__log_heartbeat_info_conditionally(log_if_no_open_orders=False)
                sleep(1800)
                self.__log_heartbeat_info_conditionally(log_if_no_open_orders=False)
                sleep(1800)
                self.__log_heartbeat_info_conditionally(log_if_no_open_orders=False)
                sleep(1800)
            except PoloniexError as e:  # TODO convert to own error type
                if str(e).find('Connection timed out.') >= 0:
                    self.logger.warning(str(e))
                else:
                    self.logger.exception('exception in heartbeat thread')
            except Exception as e:
                self.logger.exception('exception in heartbeat thread')

    def __log_heartbeat_info_conditionally(self, log_if_no_open_orders=False):
        num_open_orders = len(self.__get_pairs_with_open_orders())
        if num_open_orders == 0 and not log_if_no_open_orders:
            return
        estimated_balance = self.__get_estimated_balance()
        self.__log_heartbeat_info(estimated_balance=estimated_balance,
                                  num_open_orders=num_open_orders)

    def __log_heartbeat_info(self, estimated_balance, num_open_orders):
        self.logger.info('HEARTBEAT: estimated_balance: %f num_open_orders: %d',
                         estimated_balance, num_open_orders)

    def __get_market_date(self):
        return self.market_info.get_market_date()

    @staticmethod
    def make_order(currency, price, amount, order_type, date):
        return Order(currency=currency, price=price,
                     amount=amount, _type=order_type, date=date)

    @staticmethod
    def join(items):
        return ' '.join([str(item) for item in items])

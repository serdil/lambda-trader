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


class SidewaysStrategy:

    DELTA = 0.0001

    ORDER_TIMEOUT = 24 * 3600  # in seconds

    NUM_CHUNKS = 10
    MIN_CHUNK_SIZE = 0.00011

    HIGH_VOLUME_LIMIT = 0
    MIN_NUM_HIGH_VOLUME_PAIRS = 0

    LOOKBACK_PERIOD = 72  # in number of candlesticks

    PRICE_RANGE_LIMIT_PERCENT = 15
    PRICE_RANGE_ENTRY_RATIO = 0.1
    PRICE_RANGE_EXIT_RATIO = 0.4

    MIN_PROFIT_RATIO = 0.2

    BOUNDARY_AREA_RATIO = 0.25

    MIN_BOUNDARY_TOUCH_TO_LOOKBACK_RATIO = 0.0

    MIN_ZIG_ZAG_TO_LOOKBACK_RATIO = 0.05

    IQ_RANGE_LIMIT_PERCENT = 1
    IQ_RANGE_ENTRY_RATIO = -0.1
    IQ_RANGE_EXIT_RATIO = 0.5


    #  PENALTY_MULT = 10
    #  PENALTY_LIMIT = 10

    #  STOP_LOSS_PERCENT = 10

    def __init__(self, mult=1):

        self.__opened_orders = set()

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

                recent_candlesticks = []

                for i in range(0, self.LOOKBACK_PERIOD):
                    recent_candlesticks.append(market_info.get_pair_candlestick(pair, i))

                recent_candlesticks = list(reversed(recent_candlesticks))

                if not recent_candlesticks:
                    return

                wavg_prices = list([candlestick.weighted_average for candlestick in recent_candlesticks])
                wavg_prices.sort()

                wavg_median = wavg_prices[len(wavg_prices)//2]
                wavg_1q = wavg_prices[len(wavg_prices)//4]
                wavg_3q = wavg_prices[(len(wavg_prices)//4) * 3]

                price_min = min([candlestick.low for candlestick in recent_candlesticks])
                price_max = max([candlestick.high for candlestick in recent_candlesticks])

                price_range = price_max - price_min

                upper_area_price = price_max - price_range * self.BOUNDARY_AREA_RATIO
                lower_area_price = price_min + price_range * self.BOUNDARY_AREA_RATIO

                def touches_up(candlestick):
                    return candlestick.high >= upper_area_price

                def touches_down(candlestick):
                    return candlestick.low <= lower_area_price

                num_boundary_area_touches = 0
                num_zig_zags = 0
                last_touch_area = 'none'
                last_candlestick = recent_candlesticks[0]

                for candlestick in recent_candlesticks[1:]:
                    if touches_up(candlestick) and not touches_up(last_candlestick):
                        num_boundary_area_touches += 1
                        if last_touch_area == 'down':
                            num_zig_zags += 1
                        last_touch_area = 'up'
                    if touches_down(candlestick) and not touches_down(last_candlestick):
                        num_boundary_area_touches += 1
                        if last_touch_area == 'up':
                            num_zig_zags += 1
                        last_touch_area = 'down'
                    last_candlestick = candlestick

                boundary_touch_to_lookback_ratio = num_boundary_area_touches / self.LOOKBACK_PERIOD
                boundary_touch_condition = boundary_touch_to_lookback_ratio >= self.MIN_BOUNDARY_TOUCH_TO_LOOKBACK_RATIO

                zig_zag_to_lookback_ratio = num_zig_zags / self.LOOKBACK_PERIOD
                zig_zag_condition = zig_zag_to_lookback_ratio >= self.MIN_ZIG_ZAG_TO_LOOKBACK_RATIO

                #print(num_zig_zags)

                price_range_ratio = price_range / wavg_median
                price_range_percent = price_range_ratio * 100
                price_range_condition = price_range_percent <= self.PRICE_RANGE_LIMIT_PERCENT

                interquartile_range = wavg_3q - wavg_1q
                interquartile_range_ratio = interquartile_range / wavg_median
                interquartile_range_percent = interquartile_range_ratio * 100


                interquartile_range_condition = interquartile_range_percent <= self.IQ_RANGE_LIMIT_PERCENT

                if price_range_condition and boundary_touch_condition and zig_zag_condition:
                    current_price = market_info.get_pair_ticker(pair).lowest_ask

                    price_entry_price = price_min + price_range * self.PRICE_RANGE_ENTRY_RATIO
                    price_exit_price = price_min + price_range * self.PRICE_RANGE_EXIT_RATIO

                    iqr_entry_price = wavg_1q + interquartile_range * self.IQ_RANGE_ENTRY_RATIO
                    iqr_exit_price = wavg_1q + interquartile_range * self.IQ_RANGE_EXIT_RATIO

                    timestamp = market_info.get_market_time()

                    if current_price <= price_entry_price:

                        if price_exit_price / current_price >= self.MIN_PROFIT_RATIO:

                            print(datetime.fromtimestamp(market_info.get_market_time()))
                            account.buy(pair_second(pair), current_price, chunk_size / current_price, market_info)

                            sell_order = Order(pair_second(pair), OrderType.SELL, price_exit_price,
                                               account.get_balance(pair_second(pair)), timestamp)

                            self.__opened_orders.add(sell_order.get_order_number())

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
                if order.get_order_number() in self.__opened_orders:
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

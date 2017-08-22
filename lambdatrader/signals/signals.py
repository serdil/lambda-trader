from enum import Enum
from typing import Iterable, Optional

from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT,
    RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR,
    RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
)
from lambdatrader.models.tradesignal import (
    PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit, TradeSignal,
)


class BaseSignalGenerator:

    def __init__(self, market_info):
        self.market_info = market_info

    def generate_signals(self):
        trade_signals = self.__analyze_market()
        return trade_signals

    def __analyze_market(self):
        high_volume_pairs = self.get_allowed_pairs()
        trade_signals = list(self.__analyze_pairs(pairs=high_volume_pairs))
        return trade_signals

    def __analyze_pairs(self, pairs) -> Iterable[TradeSignal]:
        for pair in pairs:
            trade_signal = self.analyze_pair(pair=pair)
            if trade_signal:
                yield trade_signal

    def get_allowed_pairs(self):
        raise NotImplementedError

    def analyze_pair(self, pair):
        raise NotImplementedError

    def get_market_date(self):
        return self.market_info.get_market_date()


class RetracementSignalGenerator(BaseSignalGenerator):

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR
    RETRACEMENT_RATIO = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    def get_allowed_pairs(self):
        return self.__get_high_volume_pairs()

    def analyze_pair(self, pair) -> Optional[TradeSignal]:
        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.get_market_date()

        target_price = price * self.BUY_PROFIT_FACTOR
        day_high_price = latest_ticker.high24h

        price_is_lower_than_day_high = target_price < day_high_price

        if not price_is_lower_than_day_high:
            return

        current_retracement_ratio = (target_price - price) / (day_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= self.RETRACEMENT_RATIO

        if retracement_ratio_satisfied:
            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT)

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            return trade_signal

    def __get_high_volume_pairs(self):
        return sorted(
            filter(
                lambda p:
                self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                self.market_info.pairs()
            ),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )


class Direction(Enum):
    UP = 1
    DOWN = 2


class DownwardsSignalGenerator(BaseSignalGenerator):
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

    def get_allowed_pairs(self):
        return self.__get_high_volume_pairs()

    def analyze_pair(self, pair):

        try:
            old_candlestick = self.market_info.get_pair_candlestick(pair, self.LOOKBACK_PRICE_PERIOD)
            latest_ticker = self.market_info.get_pair_ticker(pair)

            old_price = old_candlestick.close
            current_price = latest_ticker.lowest_ask

            recent_volume = self.__calc_pair_recent_volume(pair, self.RECENT_VOLUME_PERIOD)

            if recent_volume == 0:
                return

            lookback_volume = self.__calc_pair_recent_volume(pair, self.LOOKBACK_VOLUME_PERIOD)
        except KeyError:
            return

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
            entry = PriceEntry(price=current_price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT)
            market_date = self.get_market_date()

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            return trade_signal

    def __calc_pair_recent_volume(self, pair, lookback):
        recent_volume = 0
        for i in range(lookback):
            recent_volume += self.market_info.get_pair_candlestick(pair, i).base_volume
        return recent_volume

    def __get_high_volume_pairs(self):
        return sorted(
            filter(
                lambda p:
                self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                self.market_info.pairs()
            ),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )

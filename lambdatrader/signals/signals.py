from logging import ERROR
from typing import Iterable, Optional

from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT, RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR, RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS,
)
from lambdatrader.loghandlers import (
    get_logger_with_all_handlers, get_logger_with_console_handler, get_silent_logger,
)
from lambdatrader.models.tradesignal import (
    PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit, TradeSignal,
)


class BaseSignalGenerator:

    def __init__(self, market_info, live=False, silent=False):
        self.market_info = market_info
        self.LIVE = live
        self.SILENT = silent

        if self.LIVE:
            self.logger = get_logger_with_all_handlers(__name__)
        elif self.SILENT:
            self.logger = get_silent_logger(__name__)
        else:
            self.logger = get_logger_with_console_handler(__name__)
            self.logger.setLevel(ERROR)

    def generate_signals(self, tracked_signals):
        self.debug('generate_signals')
        trade_signals = self.__analyze_market(tracked_signals=tracked_signals)
        return trade_signals

    def __analyze_market(self, tracked_signals):
        self.debug('__analyze_market')
        allowed_pairs = self.get_allowed_pairs()
        self.market_info.fetch_ticker()
        trade_signals = list(self.__analyze_pairs(pairs=allowed_pairs, tracked_signals=tracked_signals))
        return trade_signals

    def __analyze_pairs(self, pairs, tracked_signals) -> Iterable[TradeSignal]:
        self.debug('__analyze_pairs')
        for pair in pairs:
            trade_signal = self.analyze_pair(pair=pair, tracked_signals=tracked_signals)
            if trade_signal:
                yield trade_signal

    def get_allowed_pairs(self):
        raise NotImplementedError

    def analyze_pair(self, pair, tracked_signals):
        raise NotImplementedError

    def get_market_date(self):
        return self.market_info.get_market_date()

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def backtest_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(args)


class RetracementSignalGenerator(BaseSignalGenerator):

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR
    RETRACEMENT_RATIO = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = self.__get_high_volume_pairs()
        return high_volume_pairs

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

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
            self.debug('retracement_ratio_satisfied')
            self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
            self.debug('market_date:%s', str(market_date))
            self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
            self.debug('target_price:%s', str(target_price))
            self.debug('day_high_price:%s', str(day_high_price))

            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT)

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.debug('trade_signal:%s', str(trade_signal))

            return trade_signal

    def __get_high_volume_pairs(self):
        self.debug('__get_high_volume_pairs')
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                   self.market_info.get_active_pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )


class DynamicRetracementSignalGenerator(BaseSignalGenerator):  # TODO deduplicate logic

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR

    LOOKBACK_DRAWDOWN_RATIO = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO
    LOOKBACK_DAYS = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS

    PAIRS_RETRACEMENT_RATIOS = {}

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = self.__get_high_volume_pairs()
        return high_volume_pairs

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        try:
            self.PAIRS_RETRACEMENT_RATIOS[pair] = self.__calc_pair_retracement_ratio(pair)
        except KeyError:
            return

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.get_market_date()

        target_price = price * self.BUY_PROFIT_FACTOR
        period_high_price = self.__calc_n_days_high(pair=pair, num_days=self.LOOKBACK_DAYS)

        price_is_lower_than_period_high = target_price < period_high_price

        if not price_is_lower_than_period_high:
            return

        current_retracement_ratio = (target_price - price) / (period_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= \
                                      self.PAIRS_RETRACEMENT_RATIOS[pair]

        if retracement_ratio_satisfied:
            self.debug('retracement_ratio_satisfied')
            self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
            self.debug('market_date:%s', str(market_date))
            self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
            self.debug('target_price:%s', str(target_price))
            self.debug('day_high_price:%s', str(period_high_price))

            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT)

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.debug('trade_signal:%s', str(trade_signal))

            return trade_signal

    def __calc_pair_retracement_ratio(self, pair):
        period_max_drawdown = self.__calc_max_drawdown_since_n_days(pair, self.LOOKBACK_DAYS)

        if period_max_drawdown == 0:
            return 0.000000001

        return (self.BUY_PROFIT_FACTOR-1) / period_max_drawdown / self.LOOKBACK_DRAWDOWN_RATIO

    def __calc_max_drawdown_since_n_days(self, pair, num_days):
        lookback_num_candles = int(num_days * 24 * 3600 // 300)
        cur_max = -1
        min_since_cur_max = 1000000

        max_drawdown_max = 0
        max_drawdown_range = 0

        for i in range(lookback_num_candles - 1, -1, -1):
            candle = self.market_info.get_pair_candlestick(pair, i)

            if candle.high > cur_max:
                cur_max = candle.high
                min_since_cur_max = candle.low

            if candle.low < min_since_cur_max:
                min_since_cur_max = candle.low

            if cur_max - min_since_cur_max > max_drawdown_range:
                max_drawdown_max = cur_max
                max_drawdown_range = cur_max - min_since_cur_max

        if max_drawdown_max == 0:
            return 0

        period_max_drawdown = max_drawdown_range / max_drawdown_max
        return period_max_drawdown

    def __calc_n_days_high(self, pair, num_days):
        lookback_num_candles = int(num_days * 24 * 3600 // 300)
        high = 0

        for i in range(lookback_num_candles - 1, -1, -1):
            candle = self.market_info.get_pair_candlestick(pair, i)

            if candle.high > high:
                high = candle.high

        return high

    def __get_high_volume_pairs(self):
        self.debug('__get_high_volume_pairs')
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >=
                             self.HIGH_VOLUME_LIMIT,
                   self.market_info.get_active_pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )

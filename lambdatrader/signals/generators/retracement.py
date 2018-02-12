from typing import Optional

from lambdatrader.config import (
    RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT, RETRACEMENT_SIGNALS__ORDER_TIMEOUT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR, RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
)
from lambdatrader.models.tradesignal import (
    TradeSignal, PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit,
)
from lambdatrader.signals.constants import ONE_DAY_SECONDS
from lambdatrader.signals.generators.base import BaseSignalGenerator
from lambdatrader.signals.optimization import OptimizationMixin


class RetracementSignalGenerator(BaseSignalGenerator, OptimizationMixin):

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT

    ORDER_TIMEOUT_P1 = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR_P2 = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR
    RETRACEMENT_RATIO_P3 = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    def __init__(self, market_info, live=False, silent=False, optimize=False):
        super().__init__(market_info, live, silent)
        self.__optimize = optimize

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = self.analysis.get_high_volume_pairs(
            high_volume_limit=self.HIGH_VOLUME_LIMIT
        )
        return high_volume_pairs

    def optimization_set_params(self, *args):
        self.set_params(*args)

    def set_params(self, *args):
        self.ORDER_TIMEOUT_P1 = args[0]
        self.BUY_PROFIT_FACTOR_P2 = args[1]
        self.RETRACEMENT_RATIO_P3 = args[2]

    def optimization_get_params_info(self):
        return {
            'num_params': 3,
            'type': ['F', 'F', 'F'],
            'min': [ONE_DAY_SECONDS//24, 1.01, 0.01],
            'max': [ONE_DAY_SECONDS*7, 2.00, 0.99]
        }

    def optimization_get_optimization_periods_info(self):
        return {
            'periods': [7*ONE_DAY_SECONDS],
            'weights': [1],
            'max_costs': [10]
        }

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:
        if self.__optimize:
            self.optimization_update_parameters_if_necessary()

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.market_date

        target_price = price * self.BUY_PROFIT_FACTOR_P2
        day_high_price = latest_ticker.high24h

        price_is_lower_than_day_high = target_price < day_high_price

        if not price_is_lower_than_day_high:
            return

        current_retracement_ratio = (target_price - price) / (day_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= self.RETRACEMENT_RATIO_P3

        if retracement_ratio_satisfied:
            self.debug('retracement_ratio_satisfied')
            self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
            self.debug('market_date:%s', str(market_date))
            self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
            self.debug('target_price:%s', str(target_price))
            self.debug('day_high_price:%s', str(day_high_price))

            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT_P1)

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.debug('trade_signal:%s', str(trade_signal))

            return trade_signal

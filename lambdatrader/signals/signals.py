from typing import Iterable, Optional

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT, RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR, RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS, RED_GREEN_MARKET_NUM_PAIRS, RED_MARKET_MAJORITY_NUM,
    RED_MARKET_NUM_CANDLES, RED_MARKET_DIP_THRESHOLD,

    GREEN_MARKET_MAJORITY_NUM, GREEN_MARKET_NUM_CANDLES, GREEN_MARKET_UP_THRESHOLD,
    ENABLING_DISABLING_CHECK_INTERVAL,
)
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.loghandlers import (
    get_trading_logger,
)
from lambdatrader.models.tradesignal import (
    PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit, TradeSignal,
)
from lambdatrader.signals.analysis import Analysis
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import get_small_feature_func_set
from lambdatrader.signals.data_analysis.learning.utils import train_max_min_close_pred_lin_reg_model
from lambdatrader.signals.data_analysis.values import value_dummy
from lambdatrader.utilities.decorators import every_n_market_seconds
from lambdatrader.utilities.utils import seconds

from lambdatrader.signals.constants import ONE_DAY_SECONDS
from lambdatrader.signals.optimization import OptimizationMixin


class BaseSignalGenerator:

    def __init__(self, market_info, live=False, silent=False, **kwargs):
        self.market_info = market_info
        self.LIVE = live
        self.SILENT = silent

        self.logger = get_trading_logger(__name__, live=live, silent=silent)

        self.analysis = Analysis(market_info=market_info, live=live, silent=silent)

    def generate_signals(self, tracked_signals):
        self.debug('generate_signals')
        trade_signals = self.analyze_market(tracked_signals=tracked_signals)
        return trade_signals

    def analyze_market(self, tracked_signals):
        self.pre_analyze_market(tracked_signals)
        self.debug('analyze_market')
        allowed_pairs = self.get_allowed_pairs()
        self.market_info.fetch_ticker()
        trade_signals = list(self.analyze_pairs(pairs=allowed_pairs,
                                                tracked_signals=tracked_signals))
        self.post_analyze_market(tracked_signals)
        return trade_signals

    def analyze_pairs(self, pairs, tracked_signals) -> Iterable[TradeSignal]:
        self.debug('analyze_pairs')
        for pair in pairs:
            trade_signal = self.analyze_pair(pair=pair, tracked_signals=tracked_signals)
            if trade_signal:
                yield trade_signal

    def get_allowed_pairs(self):
        raise NotImplementedError

    def analyze_pair(self, pair, tracked_signals):
        raise NotImplementedError

    def get_market_date(self):
        return self.market_date

    @property
    def market_date(self):
        return self.market_info.market_date

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def backtest_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(*args)

    def pre_analyze_market(self, tracked_signals):
        pass

    def post_analyze_market(self, tracked_signals):
        pass


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


class DynamicRetracementSignalGenerator(BaseSignalGenerator):  # TODO deduplicate logic

    NUM_TRADING_PAIRS = 50
    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR

    LOOKBACK_DRAWDOWN_RATIO = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO
    LOOKBACK_DAYS = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS

    RED_GREEN_MARKET_NUM_PAIRS = RED_GREEN_MARKET_NUM_PAIRS

    RED_MARKET_MAJORITY_NUM = RED_MARKET_MAJORITY_NUM
    RED_MARKET_NUM_CANDLES = RED_MARKET_NUM_CANDLES
    RED_MARKET_DIP_THRESHOLD = RED_MARKET_DIP_THRESHOLD

    GREEN_MARKET_MAJORITY_NUM = GREEN_MARKET_MAJORITY_NUM
    GREEN_MARKET_NUM_CANDLES = GREEN_MARKET_NUM_CANDLES
    GREEN_MARKET_UP_THRESHOLD = GREEN_MARKET_UP_THRESHOLD

    ENABLING_DISABLING_CHECK_INTERVAL = ENABLING_DISABLING_CHECK_INTERVAL

    def __init__(self, market_info, live=False, silent=False, enable_disable=True, **kwargs):
        super().__init__(market_info, live=live, silent=silent)
        self.pairs_retracement_ratios = {}
        self.enable_disable = enable_disable
        self.trading_enabled = True
        self.last_enable_disable_checked = 0

    def get_algo_descriptor(self):
        return {
            'CLASS_NAME': self.__class__.__name__,

            'NUM_TRADING_PAIRS': self.NUM_TRADING_PAIRS,
            'HIGH_VOLUME_LIMIT': self.HIGH_VOLUME_LIMIT,
            'BUY_PROFIT_FACTOR': self.BUY_PROFIT_FACTOR,

            'LOOKBACK_DRAWDOWN_RATIO': self.LOOKBACK_DRAWDOWN_RATIO,
            'LOOKBACK_DAYS': self.LOOKBACK_DAYS,

            'RED_GREEN_MARKET_NUM_PAIRS': self.RED_GREEN_MARKET_NUM_PAIRS,

            'RED_MARKET_MAJORITY_NUM': self.RED_MARKET_MAJORITY_NUM,
            'RED_MARKET_NUM_CANDLES': self.RED_MARKET_NUM_CANDLES,
            'RED_MARKET_DIP_THRESHOLD': self.RED_MARKET_DIP_THRESHOLD,

            'GREEN_MARKET_MAJORITY_NUM': self.GREEN_MARKET_MAJORITY_NUM,
            'GREEN_MARKET_NUM_CANDLES': self.GREEN_MARKET_NUM_CANDLES,
            'GREEN_MARKET_UP_THRESHOLD': self.GREEN_MARKET_UP_THRESHOLD,
        }

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = [pair for pair
                             in self.analysis.get_high_volume_pairs(self.HIGH_VOLUME_LIMIT)
                             if pair not in ['BTC_DOGE', 'BTC_BCN']]
        high_volume_pairs = high_volume_pairs[:self.NUM_TRADING_PAIRS]
        return high_volume_pairs

    def pre_analyze_market(self, tracked_signals):
        if self.enable_disable:
            self.update_enabling_disabling_status(tracked_signals)

    @every_n_market_seconds(n=ENABLING_DISABLING_CHECK_INTERVAL)
    def update_enabling_disabling_status(self, tracked_signals):
        if self.trading_enabled:
            should_disable = self.should_disable_trading(tracked_signals=tracked_signals)
            if should_disable:
                self.logger.info('disabling_trading')
                self.backtest_print('++++++++++++++++++++++++'
                                    'DISABLING_TRADING'
                                    '++++++++++++++++++++++++')
                self.trading_enabled = False
                self.cancel_all_trades(tracked_signals)
        else:
            should_enable = self.should_enable_trading(tracked_signals=tracked_signals)
            if should_enable:
                self.logger.info('enabling_trading')
                self.backtest_print('========================'
                                    'ENABLING_TRADING'
                                    '========================')
                self.trading_enabled = True

    @staticmethod
    def cancel_all_trades(tracked_signals):
        for signal in tracked_signals:
            signal.cancel()

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        if not self.trading_enabled:
            return

        try:
            self.pairs_retracement_ratios[pair] = \
                self.analysis.calc_pair_retracement_ratio(
                    pair,
                    buy_profit_factor=self.BUY_PROFIT_FACTOR,
                    lookback_days=self.LOOKBACK_DAYS,
                    lookback_drawdown_ratio=self.LOOKBACK_DRAWDOWN_RATIO
                )
        except KeyError:
            self.logger.warning('Key error while getting candlestick for pair: %s', pair)
            return

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.market_date

        target_price = price * self.BUY_PROFIT_FACTOR
        period_high_price = self.analysis.calc_n_days_high(pair=pair, num_days=self.LOOKBACK_DAYS)

        price_is_lower_than_period_high = target_price < period_high_price

        if not price_is_lower_than_period_high:
            return

        current_retracement_ratio = (target_price - price) / (period_high_price - price)
        retracement_ratio_satisfied = (current_retracement_ratio <=
                                       self.pairs_retracement_ratios[pair])

        if retracement_ratio_satisfied:
            self.debug('retracement_ratio_satisfied')
            self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
            self.debug('market_date:%s', str(market_date))
            self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
            self.debug('target_price:%s', str(target_price))
            self.debug('day_high_price:%s', str(period_high_price))

            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=seconds(days=self.LOOKBACK_DAYS))

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.debug('trade_signal:%s', str(trade_signal))

            return trade_signal

    def should_disable_trading(self, tracked_signals):
        return self.analysis.should_stop_trading_based_on_market_red(
            num_pairs=self.RED_GREEN_MARKET_NUM_PAIRS,
            majority_num=self.RED_MARKET_MAJORITY_NUM,
            num_candles=self.RED_MARKET_NUM_CANDLES,
            dip_threshold=self.RED_MARKET_DIP_THRESHOLD
        )

    def should_enable_trading(self, tracked_signals):
        return self.analysis.should_start_trading_based_on_market_green(
            num_pairs=self.RED_GREEN_MARKET_NUM_PAIRS,
            majority_num=self.GREEN_MARKET_MAJORITY_NUM,
            num_candles=self.GREEN_MARKET_NUM_CANDLES,
            up_threshold=self.GREEN_MARKET_UP_THRESHOLD
        )


MODEL_UPDATE_INTERVAL = seconds(days=30)
TP_LEVEL = 0.9


class LinRegSignalGenerator(BaseSignalGenerator):

    NUM_CANDLES = 48
    CANDLE_PERIOD = M5
    TRAINING_LEN = seconds(days=120)
    TRAIN_VAL_RATIO = 0.6

    MAX_THR = 0.05
    CLOSE_THR = 0.04

    MAX_RMSE_THR = 0.03

    USE_RMSE_FOR_CLOSE_THR = True
    USE_RMSE_FOR_MAX_THR = True

    MIN_THR = -1.00

    def __init__(self, market_info, live=False, silent=False, **kwargs):
        super().__init__(market_info, live=live, silent=silent)
        self._dummy_market_info = BacktestingMarketInfo(candlestick_store=
                                                        CandlestickStore
                                                        .get_for_exchange(ExchangeEnum.POLONIEX))
        self.predictors = {}
        self.max_rmses = {}
        self.min_rmses = {}
        self.close_rmses = {}

    def get_algo_descriptor(self):
        return {
            'CLASS_NAME': self.__class__.__name__,
            'MODEL_UPDATE_INTERVAL': MODEL_UPDATE_INTERVAL,
            'TRAINING_LEN': self.TRAINING_LEN,
            'TRAIN_VAL_RATIO': self.TRAIN_VAL_RATIO,
            'TP_LEVEL': TP_LEVEL,
            'MAX_THR': self.MAX_THR,
            'CLOSE_THR': self.CLOSE_THR,
            'MIN_THR': self.MIN_THR,
            'USE_RMSE_FOR_MAX_THR': self.USE_RMSE_FOR_MAX_THR,
            'USE_RMSE_FOR_CLOSE_THR': self.USE_RMSE_FOR_CLOSE_THR,
            'MAX_RMSE_THR': self.MAX_RMSE_THR
        }

    def get_allowed_pairs(self):
        return sorted(self.market_info.get_active_pairs())
        # return ['BTC_LTC', 'BTC_ETH', 'BTC_ETC', 'BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC']
        # return ['BTC_XMR']
        # return ['BTC_SYS']
        # return ['BTC_ETH']
        # return ['BTC_ETC']
        # return ['BTC_VIA']
        # return ['BTC_RADS']
        # return ['BTC_XRP']
        # return ['BTC_SC']

    def pre_analyze_market(self, tracked_signals):
        self.update_predictors()

    @every_n_market_seconds(n=MODEL_UPDATE_INTERVAL)
    def update_predictors(self):
        self.logger.info('training predictors...')
        training_end_date = self.market_date - (self.NUM_CANDLES+1) * self.CANDLE_PERIOD.seconds()
        training_start_date = training_end_date - self.TRAINING_LEN
        num_pairs = len(list(self.get_allowed_pairs()))
        for i, pair in enumerate(self.get_allowed_pairs()):
            self.backtest_print('({}/{}) training: {}'.format(i, num_pairs, pair))
            self.debug('%s', '({}/{}) training: {}'.format(i, num_pairs, pair))
            try:
                train_res = train_max_min_close_pred_lin_reg_model(market_info=
                                                                   self._dummy_market_info,
                                                                   pair=pair,
                                                                   start_date=training_start_date,
                                                                   end_date=training_end_date,
                                                                   num_candles=self.NUM_CANDLES,
                                                                   candle_period=self.CANDLE_PERIOD,
                                                                   train_ratio=self.TRAIN_VAL_RATIO)
                predictor, rmses = train_res
                max_rmse, min_rmse, close_rmse = rmses
            except KeyError:
                if pair in self.predictors:
                    del self.predictors[pair]
            else:
                self.predictors[pair] = predictor
                self.max_rmses[pair] = max_rmse
                self.min_rmses[pair] = min_rmse
                self.close_rmses[pair] = close_rmse
        self.backtest_print('=================training complete!==================')
        self.logger.info('training complete!')

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

        if pair not in self.predictors:
            return

        feature_funcs = list(get_small_feature_func_set())
        value_func = value_dummy

        try:
            latest_candle = self.market_info.get_pair_candlestick(pair=pair)
            latest_candle_date = latest_candle.date

            data_set = create_pair_dataset_from_history(self._dummy_market_info,
                                                        pair=pair,
                                                        start_date=latest_candle_date,
                                                        end_date=latest_candle_date,
                                                        feature_functions=feature_funcs,
                                                        value_function=value_func)

            feature_names = data_set.get_first_feature_names()
            feature_values = data_set.get_numpy_feature_matrix()[0]
        except KeyError:
            self.logger.error('KeyError: %s', pair)
            return

        max_pred, min_pred, close_pred = self.predictors[pair](feature_names, feature_values)

        if self.USE_RMSE_FOR_MAX_THR:
            max_thr = max(self.max_rmses[pair], self.MAX_RMSE_THR)
        else:
            max_thr = self.MAX_THR

        if self.USE_RMSE_FOR_CLOSE_THR:
            close_thr = self.close_rmses[pair]
        else:
            close_thr = self.CLOSE_THR

        if max_pred < max_thr or close_pred < close_thr or min_pred < self.MIN_THR:
            return

        self.backtest_print()
        self.backtest_print(pair, 'pair preds:', max_pred, min_pred, close_pred)

        self.logger.info('signal for %s, pred values: %.4f %.4f %.4f',
                         pair, max_pred, min_pred, close_pred)

        # if self.market_date - latest_candle_date >= self.CANDLE_PERIOD.seconds():
        #     self.logger.info('latest candle out of date, dismissing signal.')
        #     return

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.market_date

        if self.LIVE:
            if price - latest_candle.close > 0.005:
                self.logger.info('lowest ask higher than latest candlestick close, '
                                 'dismissing signal.')
                return

        target_price = price * (1 + max_pred * TP_LEVEL)

        self.debug('market_date:%s', str(market_date))
        self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
        self.debug('target_price:%s', str(target_price))

        target_duration = seconds(seconds=self.NUM_CANDLES * self.CANDLE_PERIOD.seconds())

        entry = PriceEntry(price)
        success_exit = PriceTakeProfitSuccessExit(price=target_price)
        failure_exit = TimeoutStopLossFailureExit(timeout=target_duration)

        trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                   success_exit=success_exit, failure_exit=failure_exit)

        self.logger.debug('trade_signal:%s', str(trade_signal))

        return trade_signal

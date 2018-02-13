import uuid
from typing import Optional

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.models.tradesignal import (
    TradeSignal, PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit,
)
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import get_small_feature_func_set
from lambdatrader.signals.data_analysis.learning.utils import train_max_min_close_pred_lin_reg_model
from lambdatrader.signals.data_analysis.values import value_dummy
from lambdatrader.signals.generators.base import BaseSignalGenerator
from lambdatrader.signals.generators.constants import (
    LINREG__TP_STRATEGY_CLOSE_PRED_MULT, LINREG__TP_STRATEGY_MAX_PRED_MULT,
)
from lambdatrader.utilities.utils import seconds

MODEL_UPDATE_INTERVAL = seconds(days=30)
MAX_ALLOWED_TICK_CANDLE_DIFF = 0.005


class LinRegSignalGeneratorSettings:
    def __init__(self,
                 num_candles=48,
                 candle_period=M5,
                 training_len=seconds(days=500),
                 train_val_ratio=0.7,
                 max_thr=0.00,
                 close_thr=0.02,
                 min_thr=-1.00,
                 max_rmse_thr=0.03,
                 close_rmse_thr=0.02,
                 max_rmse_mult=1.0,
                 close_rmse_mult=1.0,
                 use_rmse_for_close_thr=True,
                 use_rmse_for_max_thr=False,
                 tp_level=1.0,
                 tp_strategy=LINREG__TP_STRATEGY_CLOSE_PRED_MULT,
                 exclude_external=True):
        self.num_candles = num_candles
        self.candle_period = candle_period
        self.training_len = training_len
        self.train_val_ratio = train_val_ratio
        self.max_thr = max_thr
        self.close_thr = close_thr
        self.min_thr = min_thr
        self.max_rmse_thr = max_rmse_thr
        self.close_rmse_thr = close_rmse_thr
        self.max_rmse_mult = max_rmse_mult
        self.close_rmse_mult = close_rmse_mult
        self.use_rmse_for_close_thr = use_rmse_for_close_thr
        self.use_rmse_for_max_thr = use_rmse_for_max_thr
        self.tp_level = tp_level
        self.tp_strategy = tp_strategy
        self.exclude_external = exclude_external


class LinRegSignalGenerator(BaseSignalGenerator):

    NUM_CANDLES = 48
    CANDLE_PERIOD = M5
    TRAINING_LEN = seconds(days=500)
    TRAIN_VAL_RATIO = 0.7

    MAX_THR = 0.00
    CLOSE_THR = 0.02
    MIN_THR = -1.00

    MAX_RMSE_THR = 0.03
    CLOSE_RMSE_THR = 0.02

    MAX_RMSE_MULT = 1.0
    CLOSE_RMSE_MULT = 1.0

    USE_RMSE_FOR_CLOSE_THR = True
    USE_RMSE_FOR_MAX_THR = False

    TP_LEVEL = 1.0

    TP_STRATEGY = LINREG__TP_STRATEGY_CLOSE_PRED_MULT

    EXCLUDE_EXTERNAL_SIGNAL_PAIRS = True

    def __init__(self, market_info, live=False, silent=False,
                 settings:LinRegSignalGeneratorSettings=None, **kwargs):
        super().__init__(market_info, live=live, silent=silent)
        self._dummy_market_info = BacktestingMarketInfo(candlestick_store=
                                                        CandlestickStore
                                                        .get_for_exchange(ExchangeEnum.POLONIEX))

        self.last_trained_date = 0

        self.predictors = {}
        self.max_rmses = {}
        self.min_rmses = {}
        self.close_rmses = {}

        self.id = uuid.uuid1()

        if settings is not None:
            self.NUM_CANDLES = settings.num_candles
            self.CANDLE_PERIOD = settings.candle_period
            self.TRAINING_LEN = settings.training_len
            self.TRAIN_VAL_RATIO = settings.train_val_ratio
            self.MAX_THR = settings.max_thr
            self.CLOSE_THR = settings.close_thr
            self.MIN_THR = settings.min_thr
            self.MAX_RMSE_THR = settings.max_rmse_thr
            self.CLOSE_RMSE_THR = settings.close_rmse_thr
            self.MAX_RMSE_MULT = settings.max_rmse_mult
            self.CLOSE_RMSE_MULT = settings.close_rmse_mult
            self.USE_RMSE_FOR_CLOSE_THR = settings.use_rmse_for_close_thr
            self.USE_RMSE_FOR_MAX_THR = settings.use_rmse_for_max_thr
            self.TP_LEVEL = settings.tp_level
            self.TP_STRATEGY = settings.tp_strategy
            self.EXCLUDE_EXTERNAL_SIGNAL_PAIRS = settings.exclude_external

    def get_algo_descriptor(self):
        return {
            'CLASS_NAME': self.__class__.__name__,
            'NUM_CANDLES': self.NUM_CANDLES,
            'CANDLE_PERIOD': self.CANDLE_PERIOD,
            'MODEL_UPDATE_INTERVAL': MODEL_UPDATE_INTERVAL,
            'TRAINING_LEN': self.TRAINING_LEN,
            'TRAIN_VAL_RATIO': self.TRAIN_VAL_RATIO,
            'MAX_THR': self.MAX_THR,
            'CLOSE_THR': self.CLOSE_THR,
            'MIN_THR': self.MIN_THR,
            'USE_RMSE_FOR_MAX_THR': self.USE_RMSE_FOR_MAX_THR,
            'USE_RMSE_FOR_CLOSE_THR': self.USE_RMSE_FOR_CLOSE_THR,
            'MAX_RMSE_THR': self.MAX_RMSE_THR,
            'CLOSE_RMSE_THR': self.CLOSE_RMSE_THR,
            'TP_LEVEL': self.TP_LEVEL,
            'MAX_RMSE_MULT': self.MAX_RMSE_MULT,
            'CLOSE_RMSE_MULT': self.CLOSE_RMSE_MULT,
            'TP_STRATEGY': self.TP_STRATEGY,
            'EXCLUDE_EXTERNAL_SIGNAL_PAIRS': self.EXCLUDE_EXTERNAL_SIGNAL_PAIRS,
        }

    def get_allowed_pairs(self):
        return sorted(self.market_info.get_active_pairs())
        # return ['BTC_LTC', 'BTC_ETH', 'BTC_ETC', 'BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC']
        # return ['BTC_LTC']
        # return ['BTC_XMR']
        # return ['BTC_SYS']
        # return ['BTC_ETH']
        # return ['BTC_ETC']
        # return ['BTC_VIA']
        # return ['BTC_RADS']
        # return ['BTC_XRP']
        # return ['BTC_SC']

    def pre_analyze_market(self, tracked_signals):
        market_date = self.market_date
        if market_date - self.last_trained_date > MODEL_UPDATE_INTERVAL:
            self.update_predictors()
            self.last_trained_date = market_date

    def update_predictors(self):
        self.logger.info('training predictors...')
        training_end_date = self.market_date - (self.NUM_CANDLES+1) * self.CANDLE_PERIOD.seconds()
        training_start_date = training_end_date - self.TRAINING_LEN
        num_pairs = len(list(self.get_allowed_pairs()))
        for i, pair in enumerate(self.get_allowed_pairs()):
            self.backtest_print('({}/{}) training: {}'.format(i+1, num_pairs, pair))
            self.debug('%s', '({}/{}) training: {}'.format(i+1, num_pairs, pair))
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

        if self.EXCLUDE_EXTERNAL_SIGNAL_PAIRS:
            signals_considered = tracked_signals
        else:
            signals_considered = filter(lambda s: 'origin' in s and s['origin'] == self.id,
                                        tracked_signals)

        if pair in [signal.pair for signal in signals_considered]:
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
            max_rmse = self.max_rmses[pair] * self.MAX_RMSE_MULT
            max_thr = max(max_rmse, self.MAX_RMSE_THR)
        else:
            max_thr = self.MAX_THR

        if self.USE_RMSE_FOR_CLOSE_THR:
            close_rmse = self.close_rmses[pair] * self.CLOSE_RMSE_MULT
            close_thr = max(close_rmse, self.CLOSE_RMSE_THR)
        else:
            close_thr = self.CLOSE_THR

        if max_pred < max_thr or close_pred < close_thr or min_pred < self.MIN_THR:
            return

        self.backtest_print()
        self.backtest_print(pair, 'pair preds:', max_pred, min_pred, close_pred)

        self.logger.info('signal for %s, pred values: %.4f %.4f %.4f',
                         pair, max_pred, min_pred, close_pred)

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.market_date

        if self.LIVE:
            if price > latest_candle.close * (1+MAX_ALLOWED_TICK_CANDLE_DIFF):
                self.logger.info('lowest ask higher than latest candlestick close, '
                                 'dismissing signal.')
                return

        if self.TP_STRATEGY == LINREG__TP_STRATEGY_CLOSE_PRED_MULT:
            target_price = price * (1 + close_pred * self.TP_LEVEL)
        elif self.TP_STRATEGY == LINREG__TP_STRATEGY_MAX_PRED_MULT:
            target_price = price * (1 + max_pred * self.TP_LEVEL)
        else:
            raise ValueError('invalid TP strategy')

        self.debug('market_date:%s', str(market_date))
        self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
        self.debug('target_price:%s', str(target_price))

        target_duration = seconds(seconds=self.NUM_CANDLES * self.CANDLE_PERIOD.seconds())

        entry = PriceEntry(price)
        success_exit = PriceTakeProfitSuccessExit(price=target_price)
        failure_exit = TimeoutStopLossFailureExit(timeout=target_duration)

        trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                   success_exit=success_exit, failure_exit=failure_exit,
                                   meta={'origin': self.id})

        self.logger.debug('trade_signal:%s', str(trade_signal))

        return trade_signal

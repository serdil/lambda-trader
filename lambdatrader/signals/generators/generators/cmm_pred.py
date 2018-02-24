import uuid
from collections import namedtuple
from typing import Optional

from sklearn.ensemble import RandomForestRegressor

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.models.tradesignal import (
    TradeSignal, PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit,
)
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import MaxReturn, MinReturn, CloseReturn
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory
from lambdatrader.signals.generators.constants import (
    CMM__TP_STRATEGY_CLOSE_PRED_MULT, CMM__TP_STRATEGY_MAX_PRED_MULT,
)
from lambdatrader.signals.generators.generators.base import BaseSignalGenerator
from lambdatrader.signals.generators.generators.linreg import MAX_ALLOWED_TICK_CANDLE_DIFF
from lambdatrader.utilities.utils import seconds


class CMMModelPredictorFactories:

    @classmethod
    def get_default_random_forest(cls):
        num_candles = 48
        candle_period = M5
        n_estimators = 300

        n_jobs = -1
        verbose = True

        rfr_class = RandomForestRegressor
        rfr_args = []
        rfr_kwargs = {'n_estimators': n_estimators, 'n_jobs': n_jobs, 'verbose': verbose}
        model_factory = ScikitModelFactory(model_class=rfr_class,
                                           model_args=rfr_args,
                                           model_kwargs=rfr_kwargs)
        cmm_model = CMMModel(model_factory=model_factory, num_candles=48, candle_period=M5)

        feature_set = DFFeatureSetFactory.get_all_periods_last_five_ohlcv()
        cmm_df_dataset_factory = CMMDFDatasetFactory(feature_set=feature_set,
                                                     num_candles=num_candles,
                                                     candle_period=M5)

        trainer = CMMModelPredictorFactory(feature_set=feature_set,
                                           num_candles=num_candles,
                                           candle_period=candle_period,
                                           cmm_df_dataset_factory=cmm_df_dataset_factory,
                                           cmm_model=cmm_model)
        return trainer


CloseMaxMinPred = namedtuple('CloseMaxMinPred', ['close_pred', 'max_pred', 'min_pred'])


class CMMModelPredictorFactory:
    def __init__(self, feature_set, num_candles, candle_period,
                 cmm_df_dataset_factory, cmm_model):
        self.feature_set = feature_set
        self.num_candles = num_candles
        self.candle_period = candle_period
        self.cmm_df_dataset_factory = cmm_df_dataset_factory
        self.cmm_model = cmm_model

    def get_predictor(self, cs_store, pair, start_date=None, end_date=None):
        dataset_factory = self.cmm_df_dataset_factory
        ds = dataset_factory.create_dataset(cs_store=cs_store,
                                            pair=pair,
                                            start_date=start_date,
                                            end_date=end_date)
        X = ds.feature_values
        y_close = ds.value_values(value_name=dataset_factory.y_close_name)
        y_max = ds.value_values(value_name=dataset_factory.y_max_name)
        y_min = ds.value_values(value_name=dataset_factory.y_min_name)

        model_close, model_max, model_min = self.cmm_model.train(X, y_close, y_max, y_min)
        return model_close, model_max, model_min

    def _get_predictor(self, model_close, model_max, model_min):
        def _predictor(cs_store, pair, market_date):
            lookback_days = 7

            start_date = market_date - seconds(days=lookback_days)
            end_date = market_date

            input_dataset = self.cmm_df_dataset_factory.create_dataset(cs_store=cs_store,
                                                                       pair=pair,
                                                                       start_date=start_date,
                                                                       end_date=end_date)

            close_pred = model_close.predict(input_dataset.feature_values)
            max_pred = model_max.predict(input_dataset.feature_values)
            min_pred = model_min.predict(input_dataset.feature_values)

            return close_pred, max_pred, min_pred
        return _predictor


class CMMDFDatasetFactory:
    def __init__(self, feature_set, num_candles, candle_period):
        self.feature_set = feature_set
        self.num_candles = num_candles
        self.candle_period = candle_period

        self.close_return_v = CloseReturn(self.num_candles)
        self.max_return_v = MaxReturn(self.num_candles)
        self.min_return_v = MinReturn(self.num_candles)

        self.value_set = DFFeatureSet([self.close_return_v, self.max_return_v, self.min_return_v],
                                      sort=False)

    def create_dataset(self, cs_store, pair, start_date=None, end_date=None, error_on_missing=True):
        return DFDataset.compute(pair=pair,
                                 feature_set=self.feature_set,
                                 value_set=self.value_set,
                                 start_date=start_date,
                                 end_date=end_date,
                                 cs_store=cs_store,
                                 normalize=True,
                                 error_on_missing=error_on_missing)

    @property
    def y_close_name(self):
        return self.close_return_v.name

    @property
    def y_max_name(self):
        return self.max_return_v.name

    @property
    def y_min_name(self):
        return self.y_min_name


class ScikitModelFactory:
    def __init__(self, model_class, model_args, model_kwargs):
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def create_model(self):
        return self.model_class(*self.model_args, **self.model_kwargs)


class CMMModel:

    def __init__(self, model_factory, num_candles, candle_period):
        self.model_factory = model_factory
        self.num_candles = num_candles
        self.candle_period = candle_period

    def train(self, X, y_close, y_max, y_min):
        model_close = self.model_factory.create_model()
        model_close.fit(X, y_close)

        model_max = self.model_factory.create_model()
        model_max.fit(X, y_max)

        model_min = self.model_factory.create_model()
        model_min.fit(X, y_min)

        return model_close, model_max, model_min


class CMMModelSignalGeneratorSettings:
    def __init__(self,
                 num_candles=48,
                 candle_period=M5,
                 training_len=seconds(days=500),
                 train_val_ratio=0.7,
                 max_thr=0.02,
                 close_thr=0.02,
                 min_thr=-1.00,
                 max_rmse_thr=0.02,
                 close_rmse_thr=0.02,
                 max_rmse_mult=1.0,
                 close_rmse_mult=1.0,
                 use_rmse_for_close_thr=False,
                 use_rmse_for_max_thr=False,
                 tp_level=1.0,
                 tp_strategy=CMM__TP_STRATEGY_CLOSE_PRED_MULT,
                 exclude_external=True,
                 model_update_interval=seconds(days=30),
                 cmm_model_predictor_factory=None):
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
        self.model_update_interval = model_update_interval

        if cmm_model_predictor_factory is not None:
            self.cmm_model_predictor_factory = cmm_model_predictor_factory
        else:
            raise ValueError('cmm_model_trainer cannot be empty.')


class CMMModelSignalGenerator(BaseSignalGenerator):

    def __init__(self, market_info, live=False, silent=False,
                 settings:CMMModelSignalGeneratorSettings=None,
                 cs_store=SQLiteCandlestickStore.get_for_exchange(POLONIEX),
                 **kwargs):
        super().__init__(market_info, live=live, silent=silent)
        self.cs_store = cs_store

        self.last_trained_date = 0

        self.predictors = {}
        self.id = uuid.uuid1()

        if settings is None:
            settings = CMMModelSignalGeneratorSettings(cmm_model_predictor_factory=
                                                       CMMModelPredictorFactories.get_default_random_forest())

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
        self.MODEL_UPDATE_INTERVAL = settings.model_update_interval
        self.MODEL_PREDICTOR_FACTORY = settings.cmm_model_predictor_factory

        if self.USE_RMSE_FOR_MAX_THR or self.USE_RMSE_FOR_CLOSE_THR:
            raise NotImplementedError('rmse based threshold setting not implemented.')

    def get_algo_descriptor(self):
        return {
            'CLASS_NAME': self.__class__.__name__,
            'NUM_CANDLES': self.NUM_CANDLES,
            'CANDLE_PERIOD': self.CANDLE_PERIOD,
            'MODEL_UPDATE_INTERVAL': self.MODEL_UPDATE_INTERVAL,
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
            'MODEL_PREDICTOR_FACTORY': self.MODEL_PREDICTOR_FACTORY
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
        if market_date - self.last_trained_date > self.MODEL_UPDATE_INTERVAL:
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
                predictor_factory = self.MODEL_PREDICTOR_FACTORY
                predictor = predictor_factory.get_predictor(cs_store=self.cs_store,
                                                            pair=pair,
                                                            start_date=training_start_date,
                                                            end_date=training_end_date)
            except KeyError:
                if pair in self.predictors:
                    del self.predictors[pair]
            else:
                self.predictors[pair] = predictor
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

        try:
            latest_candle = self.market_info.get_pair_candlestick(pair=pair)
            latest_candle_date = latest_candle.date

            preds = self.predictors[pair](cs_store=self.cs_store,
                                          pair=pair,
                                          market_date=latest_candle_date)
            close_pred, max_pred, min_pred = preds.close_pred, preds.max_pred, preds.min_pred
        except KeyError:
            self.logger.error('KeyError: %s', pair)
            return

        max_thr = self.MAX_THR

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

        if self.TP_STRATEGY == CMM__TP_STRATEGY_CLOSE_PRED_MULT:
            target_price = price * (1 + close_pred * self.TP_LEVEL)
        elif self.TP_STRATEGY == CMM__TP_STRATEGY_MAX_PRED_MULT:
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

from collections import namedtuple

import pandas as pd
from pandas.core.base import DataError
from sklearn.ensemble import RandomForestRegressor

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseReturn, MaxReturn, MinReturn
from lambdatrader.signals.data_analysis.factories import FeatureSets
from lambdatrader.signals.generators.constants import (
    LINREG__TP_STRATEGY_MAX_PRED_MULT, LINREG__TP_STRATEGY_CLOSE_PRED_MULT,
    CMM__TP_STRATEGY_MAX_PRED_MULT,
)
from lambdatrader.signals.generators.generators.cmm_pred import (
    CMMModelSignalGeneratorSettings, CMMModelSignalGenerator,
)
from lambdatrader.signals.generators.cmm_models import SklearnCMMModel, XGBCMMModel
from lambdatrader.signals.generators.generators.linreg import (
    LinRegSignalGeneratorSettings, LinRegSignalGenerator,
)
from lambdatrader.utilities.utils import seconds


def get_lin_reg_first_conf_settings(exclude_external=True):
    return LinRegSignalGeneratorSettings(
                 num_candles=48,
                 candle_period=M5,
                 training_len=seconds(days=120),
                 train_val_ratio=0.6,
                 max_thr=0.05,
                 close_thr=0.04,
                 min_thr=-1.00,
                 max_rmse_thr=0.00,
                 close_rmse_thr=0.00,
                 max_rmse_mult=1.0,
                 close_rmse_mult=1.0,
                 use_rmse_for_close_thr=False,
                 use_rmse_for_max_thr=False,
                 tp_level=1.0,
                 tp_strategy=LINREG__TP_STRATEGY_MAX_PRED_MULT,
                 exclude_external=exclude_external)


def get_lin_reg_second_conf_settings(exclude_external=True):
    return LinRegSignalGeneratorSettings(
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
                 exclude_external=exclude_external)


class LinRegSignalGeneratorFactory:

    def __init__(self, market_info, live=False, silent=False):
        self.market_info = market_info
        self.live = live
        self.silent = silent

    def get_excluding_first_conf_lin_reg_signal_generator(self):
        settings = get_lin_reg_first_conf_settings(exclude_external=True)
        return self._create_with_settings(settings)

    def get_excluding_second_conf_lin_reg_signal_generator(self):
        settings = get_lin_reg_second_conf_settings(exclude_external=True)
        return self._create_with_settings(settings)

    def get_non_excluding_first_conf_lin_reg_signal_generator(self):
        settings = get_lin_reg_first_conf_settings(exclude_external=False)
        return self._create_with_settings(settings)

    def get_non_excluding_second_conf_lin_reg_signal_generator(self):
        settings = get_lin_reg_second_conf_settings(exclude_external=False)
        return self._create_with_settings(settings)

    def _create_with_settings(self, settings):
        return LinRegSignalGenerator(self.market_info,
                                     live=self.live, silent=self.silent, settings=settings)


class CMMModelPredictorFactoryFactory:

    def __init__(self,
                 precompute=False,
                 pc_pairs=None,
                 pc_start_date=None,
                 pc_end_date=None,
                 pc_cs_store=None):
        self.precompute = precompute
        self.pc_pairs = pc_pairs
        self.pc_start_date = pc_start_date
        self.pc_end_date = pc_end_date
        self.pc_cs_store = pc_cs_store

    def get_default_random_forest(self, n_estimators=300):
        num_candles = 48
        candle_period = M5
        n_estimators = n_estimators

        n_jobs = 1
        verbose = False

        rfr_class = RandomForestRegressor
        rfr_args = []
        rfr_kwargs = {'n_estimators': n_estimators, 'n_jobs': n_jobs, 'verbose': verbose}
        model_factory = ScikitModelFactory(model_class=rfr_class,
                                           model_args=rfr_args,
                                           model_kwargs=rfr_kwargs)
        cmm_model = SklearnCMMModel(model_factory=model_factory, num_candles=48, candle_period=M5)

        feature_set = FeatureSets.get_all_periods_last_five_ohlcv()
        cmm_df_dataset_factory = CMMDFDatasetFactory(feature_set=feature_set,
                                                     num_candles=num_candles,
                                                     candle_period=M5)

        predictor_factory = CMMModelPredictorFactory(feature_set=feature_set,
                                                     num_candles=num_candles,
                                                     candle_period=candle_period,
                                                     cmm_df_dataset_factory=cmm_df_dataset_factory,
                                                     cmm_model=cmm_model,
                                                     precompute=self.precompute,
                                                     pc_pairs=self.pc_pairs,
                                                     pc_start_date=self.pc_start_date,
                                                     pc_end_date=self.pc_end_date,
                                                     pc_cs_store=self.pc_cs_store)
        return predictor_factory

    def get_xgb_lin_reg(self):
        num_candles = 48
        candle_period = M5

        train_val_ratio = 0.9
        n_rounds = 1000
        early_stopping_rounds = 10

        booster_params = self._xgb_lin_reg_params_2()

        cmm_model = XGBCMMModel(num_candles=num_candles,
                                candle_period=M5,
                                booster_params=booster_params,
                                train_ratio=train_val_ratio,
                                num_rounds=n_rounds,
                                early_stopping_rounds=early_stopping_rounds)

        feature_set = FeatureSets.get_all_periods_last_ten_ohlcv()
        cmm_df_dataset_factory = CMMDFDatasetFactory(feature_set=feature_set,
                                                     num_candles=num_candles, candle_period=M5)

        predictor_factory = CMMModelPredictorFactory(feature_set=feature_set,
                                                     num_candles=num_candles,
                                                     candle_period=candle_period,
                                                     cmm_df_dataset_factory=cmm_df_dataset_factory,
                                                     cmm_model=cmm_model,
                                                     precompute=self.precompute,
                                                     pc_pairs=self.pc_pairs,
                                                     pc_start_date=self.pc_start_date,
                                                     pc_end_date=self.pc_end_date,
                                                     pc_cs_store=self.pc_cs_store)
        return predictor_factory

    @staticmethod
    def _xgb_lin_reg_params_1():
        booster_params = {
            'silent': 1,
            'booster': 'gblinear',

            'objective': 'reg:linear',
            'base_score': 0,
            'eval_metric': 'rmse',

            'eta': 0.03,
            'gamma': 0,
            'max_depth': 3,
            'min_child_weight': 2,
            'max_delta_step': 0,
            'subsample': 1,
            'colsample_bytree': 1,
            'colsample_bylevel': 1,
            'lambda': 0,
            'alpha': 0,
            'tree_method': 'auto',
            'sketch_eps': 0.03,
            'scale_pos_weight': 1,
            'updater': 'grow_colmaker,prune',
            'refresh_leaf': 1,
            'process_type': 'default',
            'grow_policy': 'depthwise',
            'max_leaves': 0,
            'max_bin': 256,

            'sample_type': 'weighted',
            'rate_drop': 0.01,
        }
        return booster_params

    @staticmethod
    def _xgb_lin_reg_params_2():
        booster_params = {
            'silent': 1, 'booster': 'gblinear',

            'objective': 'reg:linear', 'base_score': 0, 'eval_metric': 'rmse',

            'eta': 0.1, 'gamma': 0, 'max_depth': 3, 'min_child_weight': 1, 'max_delta_step': 0,
            'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'tree_method': 'exact',
            'sketch_eps': 0.03, 'scale_pos_weight': 1, 'refresh_leaf': 1, 'process_type': 'default',
            'grow_policy': 'depthwise', 'max_leaves': 0, 'max_bin': 256,

            # 'lambda': 1,
            # 'alpha': 0,
            # 'updater': 'grow_colmaker,prune',

            # 'sample_type': 'uniform',
            # 'normalize_type': 'tree',
            # 'rate_drop': 0.00,
            # 'one_drop': 0,
            # 'skip_drop': 0.00,

            'reg_lambda': 0, 'reg_alpha': 0,
            # 'updater': 'shotgun'
        }

        return booster_params


CloseMaxMinPred = namedtuple('CloseMaxMinPred', ['close_pred', 'max_pred', 'min_pred'])


class CMMModelPredictorFactory:

    def __init__(self,
                 feature_set,
                 num_candles,
                 candle_period,
                 cmm_df_dataset_factory,
                 cmm_model,
                 precompute=False,
                 pc_pairs=None,
                 pc_start_date=None,
                 pc_end_date=None,
                 pc_cs_store=None):
        self.feature_set = feature_set
        self.num_candles = num_candles
        self.candle_period = candle_period
        self.cmm_df_dataset_factory = cmm_df_dataset_factory
        self.cmm_model = cmm_model
        self.precompute = precompute

        if precompute:
            self.cmm_df_dataset_factory.precompute_for_pairs(cs_store=pc_cs_store,
                                                             pairs=pc_pairs,
                                                             start_date=pc_start_date,
                                                             end_date=pc_end_date)

    def get_predictor(self, cs_store, pair, start_date=None, end_date=None):
        d_f = self.cmm_df_dataset_factory
        ds = d_f.create_dataset(cs_store=cs_store,
                                pair=pair,
                                start_date=start_date,
                                end_date=end_date)
        return self._predictor_from_dataset(ds,
                                            y_close_name=d_f.y_close_name,
                                            y_max_name=d_f.y_max_name,
                                            y_min_name=d_f.y_min_name)
    
    def get_predictor_interleaved(self, cs_store, pairs, start_date=None, end_date=None):
        d_f = self.cmm_df_dataset_factory
        ds = d_f.create_interleaved(cs_store=cs_store,
                                    pairs=pairs,
                                    start_date=start_date,
                                    end_date=end_date)
        return self._predictor_from_dataset(ds,
                                            y_close_name=d_f.y_close_name,
                                            y_max_name=d_f.y_max_name,
                                            y_min_name=d_f.y_min_name)

    def _predictor_from_dataset(self, ds, y_close_name, y_max_name, y_min_name):
        X = ds.get_feature_values()
        y_close = ds.get_value_values(value_name=y_close_name)
        y_max = ds.get_value_values(value_name=y_max_name)
        y_min = ds.get_value_values(value_name=y_min_name)

        model_close, model_max, model_min = self.cmm_model.train(X, y_close, y_max, y_min)
        return self._get_predictor(model_close=model_close, model_max=model_max,
                                   model_min=model_min)

    def _get_predictor(self, model_close, model_max, model_min):
        def _predictor(cs_store, pair, date):
            # start = time.time()

            if self.precompute:
                input_dataset = self.cmm_df_dataset_factory.get_precomputed(pair)
            else:
                start_date = date
                end_date = date

                input_dataset = self.cmm_df_dataset_factory.create_dataset(cs_store=cs_store,
                                                                           pair=pair,
                                                                           start_date=start_date,
                                                                           end_date=end_date)
            # print('dataset get/creation:', time.time() - start)
            # start = time.time()

            feature_row = input_dataset.get_feature_row(pd.Timestamp(date, unit='s'))

            # print('row get:', time.time() - start)
            # start = time.time()

            close_pred = model_close.predict(feature_row)[-1]
            max_pred = model_max.predict(feature_row)[-1]
            min_pred = model_min.predict(feature_row)[-1]

            # print('prediction:', time.time() - start)

            return CloseMaxMinPred(close_pred=close_pred, max_pred=max_pred, min_pred=min_pred)
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

        self._precomputed = {}

    def precompute_for_pair(self, cs_store, pair, start_date, end_date):
        self._precomputed[pair] = self.create_dataset(cs_store=cs_store,
                                                      pair=pair,
                                                      start_date=start_date,
                                                      end_date=end_date,
                                                      error_on_missing=False)

    def precompute_for_pairs(self, cs_store, pairs, start_date, end_date):
        for pair in pairs:
            try:
                self.precompute_for_pair(cs_store, pair, start_date, end_date)
            except DataError:
                print('DataError while precomputing for pair: {}', pair)

    def create_dataset(self, cs_store, pair, start_date=None, end_date=None, error_on_missing=True):
        return DFDataset.compute(pair=pair,
                                 feature_set=self.feature_set,
                                 value_set=self.value_set,
                                 start_date=start_date,
                                 end_date=end_date,
                                 cs_store=cs_store,
                                 normalize=True,
                                 error_on_missing=error_on_missing)

    def create_interleaved(self, cs_store, pairs, start_date, end_date, error_on_missing=False):
        return DFDataset.compute_interleaved(pairs=pairs,
                                             feature_set=self.feature_set,
                                             value_set=self.value_set,
                                             start_date=start_date,
                                             end_date=end_date,
                                             cs_store=cs_store,
                                             normalize=True,
                                             error_on_missing=error_on_missing)

    def get_precomputed(self, pair):
        return self._precomputed[pair]

    @property
    def y_close_name(self):
        return self.close_return_v.name

    @property
    def y_max_name(self):
        return self.max_return_v.name

    @property
    def y_min_name(self):
        return self.min_return_v.name


class ScikitModelFactory:
    def __init__(self, model_class, model_args, model_kwargs):
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def create_model(self):
        return self.model_class(*self.model_args, **self.model_kwargs)


class CMMModelSignalGeneratorFactory:

    def __init__(self, cs_store, market_info, live=False, silent=False, pairs=None,
                 precompute=False, pc_start_date=None, pc_end_date=None):
        self.cs_store = cs_store
        self.market_info = market_info
        self.live = live
        self.silent = silent
        self.pairs = pairs

        self.precompute = precompute
        self.pc_start_date = pc_start_date
        self.pc_end_date = pc_end_date

    def get_random_forest_n_days_n_estimators(self, n_days=500, n_estimators=300):
        predictor_fact_fact = CMMModelPredictorFactoryFactory(precompute=self.precompute,
                                                              pc_pairs=self.pairs,
                                                              pc_start_date=self.pc_start_date,
                                                              pc_end_date=self.pc_end_date,
                                                              pc_cs_store=self.cs_store)
        predictor_factory = predictor_fact_fact.get_default_random_forest(n_estimators=n_estimators)
        settings = CMMModelSignalGeneratorSettings(training_len=seconds(days=n_days),
                                                   cmm_model_predictor_factory=predictor_factory)
        return self._create_with_settings(settings)

    def get_xgb_lin_reg_n_days(self, n_days):
        predictor_fact_fact = CMMModelPredictorFactoryFactory(precompute=self.precompute,
                                                              pc_pairs=self.pairs,
                                                              pc_start_date=self.pc_start_date,
                                                              pc_end_date=self.pc_end_date,
                                                              pc_cs_store=self.cs_store)
        predictor_factory = predictor_fact_fact.get_xgb_lin_reg()
        settings = CMMModelSignalGeneratorSettings(training_len=seconds(days=n_days),
                                                   cmm_model_predictor_factory=predictor_factory)
        return self._create_with_settings(settings)

    def get_xgb_lin_reg_n_days_one_model(self,
                                         n_days=200,
                                         training_pairs=None,
                                         close_thr=0.03,
                                         retrain_interval_days=30,
                                         tp_strategy=CMM__TP_STRATEGY_MAX_PRED_MULT):
        if training_pairs is None:
            training_pairs = Pairs.all_pairs()

        predictor_fact_fact = CMMModelPredictorFactoryFactory(precompute=self.precompute,
                                                              pc_pairs=self.pairs,
                                                              pc_start_date=self.pc_start_date,
                                                              pc_end_date=self.pc_end_date,
                                                              pc_cs_store=self.cs_store)
        predictor_factory = predictor_fact_fact.get_xgb_lin_reg()

        model_update_interval = seconds(days=retrain_interval_days)
        settings = CMMModelSignalGeneratorSettings(training_len=seconds(days=n_days),
                                                   cmm_model_predictor_factory=predictor_factory,
                                                   one_model_to_rule_them_all=True,
                                                   one_model_training_pairs=training_pairs,
                                                   tp_strategy=tp_strategy,
                                                   close_thr=close_thr,
                                                   model_update_interval=model_update_interval)
        return self._create_with_settings(settings)

    def _create_with_settings(self, settings):
        return CMMModelSignalGenerator(market_info=self.market_info,
                                       live=self.live, silent=self.silent,
                                       settings=settings, cs_store=self.cs_store, pairs=self.pairs)


class Pairs:

    @classmethod
    def all_pairs(cls, cs_store=SQLiteCandlestickStore.get_for_exchange(POLONIEX)):
        return sorted(cs_store.get_pairs())

    @classmethod
    def n_pairs(cls, n=None):
        pair_list = ['BTC_LTC', 'BTC_ETH', 'BTC_ETC', 'BTC_XMR',
                     'BTC_SYS', 'BTC_VIA', 'BTC_SC', 'BTC_RADS', 'BTC_RIC', 'BTC_XRP',]
        if n:
            return pair_list[:n]
        else:
            return pair_list

    @classmethod
    def eth(cls):
        return ['BTC_ETH']

    @classmethod
    def ltc(cls):
        return ['BTC_LTC']

    @classmethod
    def xmr(cls):
        return ['BTC_XMR']

    @classmethod
    def sys(cls):
        return ['BTC_SYS']

    @classmethod
    def etc(cls):
        return ['BTC_ETC']

    @classmethod
    def via(cls):
        return ['BTC_VIA']

    @classmethod
    def rads(cls):
        return ['BTC_RADS']

    @classmethod
    def ric(cls):
        return ['BTC_RIC']

    @classmethod
    def xrp(cls):
        return ['BTC_XRP']

from collections import namedtuple

from sklearn.ensemble import RandomForestRegressor

from lambdatrader.constants import M5
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseReturn, MaxReturn, MinReturn
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory
from lambdatrader.signals.generators.constants import (
    LINREG__TP_STRATEGY_MAX_PRED_MULT, LINREG__TP_STRATEGY_CLOSE_PRED_MULT,
)
from lambdatrader.signals.generators.generators.cmm_pred import (
    CMMModel, CMMModelSignalGeneratorSettings, CMMModelSignalGenerator,
)
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

        predictor_factory = CMMModelPredictorFactory(feature_set=feature_set,
                                                     num_candles=num_candles,
                                                     candle_period=candle_period,
                                                     cmm_df_dataset_factory=cmm_df_dataset_factory,
                                                     cmm_model=cmm_model)
        return predictor_factory


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
        y_close = ds.get_value_values(value_name=dataset_factory.y_close_name)
        y_max = ds.get_value_values(value_name=dataset_factory.y_max_name)
        y_min = ds.get_value_values(value_name=dataset_factory.y_min_name)

        model_close, model_max, model_min = self.cmm_model.train(X, y_close, y_max, y_min)
        return self._get_predictor(model_close=model_close,
                                   model_max=model_max,
                                   model_min=model_min)

    def _get_predictor(self, model_close, model_max, model_min):
        def _predictor(cs_store, pair, market_date):
            lookback_days = 7

            start_date = market_date - seconds(days=lookback_days)
            end_date = market_date

            input_dataset = self.cmm_df_dataset_factory.create_dataset(cs_store=cs_store,
                                                                       pair=pair,
                                                                       start_date=start_date,
                                                                       end_date=end_date)

            close_pred = model_close.predict(input_dataset.feature_values[-1].reshape(1, -1))[-1]
            max_pred = model_max.predict(input_dataset.feature_values[-1].reshape(1, -1))[-1]
            min_pred = model_min.predict(input_dataset.feature_values[-1].reshape(1, -1))[-1]

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
        return self.min_return_v.name


class ScikitModelFactory:
    def __init__(self, model_class, model_args, model_kwargs):
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def create_model(self):
        return self.model_class(*self.model_args, **self.model_kwargs)


class CMMModelSignalGeneratorFactory:

    def __init__(self, cs_store, market_info, live=False, silent=False):
        self.cs_store = cs_store
        self.market_info = market_info
        self.live = live
        self.silent = silent

    def get_random_forest_n_days(self, n=500):
        predictor_factory = CMMModelPredictorFactories.get_default_random_forest()
        settings = CMMModelSignalGeneratorSettings(training_len=seconds(days=n),
                                                   cmm_model_predictor_factory=predictor_factory)
        return self._create_with_settings(settings)

    def _create_with_settings(self, settings):
        return CMMModelSignalGenerator(market_info=self.market_info,
                                       live=self.live, silent=self.silent,
                                       settings=settings, cs_store=self.cs_store)

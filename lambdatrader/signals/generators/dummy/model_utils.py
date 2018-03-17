import random
from typing import List

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import (
    DateRange, SplitDateRange, DatasetDescriptor, SplitDatasetDescriptor,
)
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets
from lambdatrader.signals.data_analysis.models import XGBSplitDatasetModel
from lambdatrader.signals.generators.dummy.backtest_util import do_backtest
from lambdatrader.signals.generators.dummy.feature_spaces import all_sampler
from lambdatrader.signals.generators.dummy.signal_generation import (
    CloseAvgReturnMaxReturnSignalConverter, SignalServer, ModelPredSignalGenerator,
)
from lambdatrader.signals.generators.factories import Pairs

DEFAULT_XGB_BOOSTER_PARAMS = {
    'silent': 1,
    'booster': 'gbtree',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.03,
    'gamma':0,
    'max_depth': 4,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'tree_method': 'exact',
    'sketch_eps': 0.03,
    'scale_pos_weight': 1,
    'refresh_leaf': 1,
    'process_type': 'default',
    'grow_policy': 'depthwise',
    'max_leaves': 0,
    'max_bin': 256,

    'lambda': 1,
    'alpha': 0,
    'updater': 'grow_colmaker,prune',

    # 'sample_type': 'uniform',
    # 'normalize_type': 'tree',
    # 'rate_drop': 0.00,
    # 'one_drop': 0,
    # 'skip_drop': 0.00,

    # 'reg_lambda': 0,
    # 'reg_alpha': 0,
    # 'updater': 'shotgun'
}


class LearningTask:
    FEAT_SEL_SHR = 'shrinking'
    FEAT_SEL_GROW_SHR = 'growing_shrinking'
    FEAT_SEL_HIER = 'hierarchical'

    MODEL_XGB = 'xgb'
    MODEL_RDT = 'rdt'

    def __init__(self):
        self.model_per_pair = None
        self.train_pairs_interleaved = None
        self.train_pairs = None
        self.val_pairs = None
        self.test_pairs = None
        self.train_date_range = None
        self.val_date_range = None
        self.test_date_range = None
        self.feature_set = None
        self.n_candles = None

        self.close_value_set = None
        self.max_value_set = None

        self.c_thr = None
        self.m_thr = None

        self.model_to_use = None

        self.xgb_booster_params = None
        self.xgb_n_rounds = None
        self.xgb_esr = None

        self.select_features = None
        self.feat_sel_mode = None
        self.n_target_feat = None
        self.sel_ratio = None
        self.feat_sel_n_rounds = None

        self.apply_defaults()

        self.feat_sampler = all_sampler

    def apply_defaults(self):
        self.set_model_per_pair(False)
        self.set_train_pairs_interleaved(False)
        self.set_train_pairs(Pairs.eth())
        self.set_val_pairs(Pairs.eth())
        self.set_test_pairs(Pairs.eth())
        self.set_train_val_test_date_ranges(SplitDateRanges
                                            .january_20_days_test_20_days_val_160_days_train())
        self.set_feature_set(all_sampler.sample(size=100))
        self.set_n_candles(48)
        self.set_c_thr(0.01)
        self.set_m_thr(0.02)

        self.use_xgb_model()
        self.set_xgb_booster_params(DEFAULT_XGB_BOOSTER_PARAMS)
        self.set_xgb_n_rounds(1000)
        self.set_xgb_esr(10)

        return self

    def set_n_candles(self, n_candles):
        self.n_candles = n_candles
        self.close_value_set = DFFeatureSet(features=[CloseAvgReturn(n_candles=self.n_candles)])
        self.max_value_set = DFFeatureSet(features=[MaxReturn(n_candles=self.n_candles)])
        return self

    def set_model_per_pair(self, model_per_pair: bool):
        self.model_per_pair = model_per_pair
        return self

    def set_train_pairs_interleaved(self, interleaved: bool):
        self.train_pairs_interleaved = True
        return self

    def set_train_pairs(self, training_pairs: List[str]):
        self.train_pairs = training_pairs
        return self

    def set_val_pairs(self, val_pairs: List[str]):
        self.val_pairs = val_pairs
        return self

    def set_test_pairs(self, test_pairs: List[str]):
        self.test_pairs = test_pairs
        return self

    def set_train_date_range(self, date_range: DateRange):
        self.train_date_range = date_range
        return self

    def set_val_date_range(self, date_range: DateRange):
        self.val_date_range = date_range
        return self

    def set_test_date_range(self, date_range: DateRange):
        self.test_date_range = date_range
        return self

    def set_train_val_test_date_ranges(self, date_ranges: SplitDateRange):
        self.set_train_date_range(date_ranges.training)
        self.set_val_date_range(date_ranges.validation)
        self.set_test_date_range(date_ranges.test)
        return self

    def set_feature_set(self, feature_set):
        self.feature_set = feature_set
        return self

    def set_c_thr(self, c_thr):
        self.c_thr = c_thr
        return self

    def set_m_thr(self, m_thr):
        self.m_thr = m_thr
        return self

    def set_shrinking_feat_sel(self):
        self.select_features = True
        self.feat_sel_mode = self.FEAT_SEL_SHR
        return self

    def set_grow_shr_feat_sel(self):
        self.select_features = True
        self.feat_sel_mode = self.FEAT_SEL_GROW_SHR
        return self

    def set_hier_feat_sel(self):
        self.select_features = True
        self.feat_sel_mode = self.FEAT_SEL_HIER
        return self

    def set_feat_sel_n_target_feat(self, n_target_feat):
        self.n_target_feat = n_target_feat
        return self

    def set_feat_sel_sel_ratio(self, sel_ratio):
        self.sel_ratio = sel_ratio
        return self

    def set_feat_sel_n_rounds(self, feat_sel_n_rounds):
        self.feat_sel_n_rounds = feat_sel_n_rounds
        return self

    def use_xgb_model(self):
        self.model_to_use = self.MODEL_XGB
        return self

    def set_xgb_booster_params(self, booster_params):
        self.xgb_booster_params = booster_params
        return self

    def set_xgb_n_rounds(self, n_rounds):
        self.xgb_n_rounds = n_rounds
        return self

    def set_xgb_esr(self, early_stopping_rounds):
        self.xgb_esr = early_stopping_rounds
        return self

    def set_rdt_model(self):
        self.model_to_use = self.MODEL_RDT
        return self

    def execute(self):
        random.seed(0)

        if not self.train_pairs_interleaved or self.model_per_pair:
            raise NotImplementedError

        c_train_dataset = DatasetDescriptor(pairs=self.train_pairs,
                                            feature_set=self.feature_set,
                                            value_set=self.close_value_set,
                                            start_date=self.train_date_range.start,
                                            end_date=self.train_date_range.end,
                                            exchanges=(POLONIEX,),
                                            interleaved=True)
        c_val_dataset = DatasetDescriptor(pairs=self.val_pairs,
                                          feature_set=self.feature_set,
                                          value_set=self.close_value_set,
                                          start_date=self.train_date_range.start,
                                          end_date=self.train_date_range.end,
                                          exchanges=(POLONIEX,),
                                          interleaved=True)
        c_test_dataset = DatasetDescriptor(pairs=self.test_pairs,
                                           feature_set=self.feature_set,
                                           value_set=self.close_value_set,
                                           start_date=self.train_date_range.start,
                                           end_date=self.train_date_range.end,
                                           exchanges=(POLONIEX,), interleaved=True)

        c_dataset = SplitDatasetDescriptor(c_train_dataset, c_val_dataset, c_test_dataset)

        m_train_dataset = DatasetDescriptor(pairs=self.train_pairs,
                                            feature_set=self.feature_set,
                                            value_set=self.max_value_set,
                                            start_date=self.train_date_range.start,
                                            end_date=self.train_date_range.end,
                                            exchanges=(POLONIEX,),
                                            interleaved=True)
        m_val_dataset = DatasetDescriptor(pairs=self.val_pairs,
                                          feature_set=self.feature_set,
                                          value_set=self.max_value_set,
                                          start_date=self.train_date_range.start,
                                          end_date=self.train_date_range.end,
                                          exchanges=(POLONIEX,),
                                          interleaved=True)
        m_test_dataset = DatasetDescriptor(pairs=self.test_pairs,
                                           feature_set=self.feature_set,
                                           value_set=self.max_value_set,
                                           start_date=self.train_date_range.start,
                                           end_date=self.train_date_range.end,
                                           exchanges=(POLONIEX,), interleaved=True)

        m_dataset = SplitDatasetDescriptor(m_train_dataset, m_val_dataset, m_test_dataset)

        if self.model_to_use == self.MODEL_XGB:
            close_model  = XGBSplitDatasetModel(
                dataset_descriptor=c_dataset,
                booster_params=self.xgb_booster_params,
                num_round=self.xgb_n_rounds,
                early_stopping_rounds=self.xgb_esr,
                obj_name='close'
            )

            max_model = XGBSplitDatasetModel(
                dataset_descriptor=m_dataset,
                booster_params=self.xgb_booster_params,
                num_round=self.xgb_n_rounds,
                early_stopping_rounds=self.xgb_esr,
                obj_name='max'
            )
        else:
            raise NotImplementedError

        if self.select_features:
            fs = FeatureSets
            if self.feat_sel_mode == self.FEAT_SEL_SHR:
                close_model.train()
                for i in range(self.feat_sel_n_rounds):
                    selected_features = close_model.select_features_by_ratio(self.sel_ratio)
                    print('close round {} num_features:'
                          .format(i), len(selected_features.features))
                    close_model = self._replace_model_feature_set(close_model, selected_features)
                    close_model.train()
                max_model.train()
                for i in range(self.feat_sel_n_rounds):
                    selected_features = max_model.select_features_by_ratio(self.sel_ratio)
                    print('close round {} num_features:'
                          .format(i), len(selected_features.features))
                    max_model = self._replace_model_feature_set(max_model, selected_features)
                    max_model.train()
            elif self.feat_sel_mode == self.FEAT_SEL_GROW_SHR:
                self._bprint('[close] initial model')
                close_model.train()
                for i in range(self.feat_sel_n_rounds):
                    cur_features = self._get_model_feature_set(close_model)
                    cur_num_features = cur_features.num_features
                    num_new_features = int(self.n_target_feat / self.sel_ratio) - cur_num_features
                    new_features = self.feat_sampler.sample(size=num_new_features)
                    combined_features = fs.compose_remove_duplicates(cur_features, new_features)
                    close_model = self._replace_model_feature_set(close_model, combined_features)
                    self._bprint('[close] selection round')
                    close_model.train()
                    selected_features = close_model.select_features_by_number(self.n_target_feat)
                    close_model = self._replace_model_feature_set(close_model, selected_features)
                    self._bprint('[close] round {} result (num_features: {})'
                                 .format(i, selected_features.num_features))
                    close_model.train()
                max_model.train()
                for i in range(self.feat_sel_n_rounds):
                    cur_features = self._get_model_feature_set(max_model)
                    cur_num_features = cur_features.num_features
                    num_new_features = int(self.n_target_feat / self.sel_ratio) - cur_num_features
                    new_features = self.feat_sampler.sample(size=num_new_features)
                    combined_features = fs.compose_remove_duplicates(cur_features, new_features)
                    max_model = self._replace_model_feature_set(max_model, combined_features)
                    self._bprint('[max] selection round')
                    max_model.train()
                    selected_features = max_model.select_features_by_number(self.n_target_feat)
                    max_model = self._replace_model_feature_set(max_model, selected_features)
                    self._bprint('[max] round {} result (num_features: {})'
                                 .format(i, selected_features.num_features))
                    max_model.train()
            else:
                raise NotImplementedError

            signal_converter = CloseAvgReturnMaxReturnSignalConverter(c_thr=self.c_thr,
                                                                      m_thr=self.m_thr,
                                                                      n_candles=self.n_candles)

            models = [close_model, max_model]

            pairs = self.test_pairs

            start_date = self.test_date_range.start + self.n_candles * M5.seconds()
            end_date = self.test_date_range.end
            model_per_pair = self.model_per_pair
            pair_models = {}
            models = models

            signal_server = SignalServer(models=models, signal_converter=signal_converter,
                                         pairs=self.test_pairs, pc_start_date=start_date,
                                         pc_end_date=end_date, model_per_pair=model_per_pair,
                                         pair_models=pair_models)

            cs_store = ChunkCachingCandlestickStore.get_for_exchange(POLONIEX)

            market_info = BacktestingMarketInfo(candlestick_store=cs_store)

            signal_generator = ModelPredSignalGenerator(market_info=market_info,
                                                        signal_server=signal_server, pairs=pairs)

            do_backtest(signal_generator, market_info, start_date, end_date)

    @classmethod
    def _get_model_feature_set(cls, model):
        return model.dataset_descriptor.training.feature_set

    @classmethod
    def _replace_model_feature_set(cls, model, feature_set):
        model.dataset_descriptor.training.feature_set = feature_set
        model.dataset_descriptor.validation.feature_set = feature_set
        model.dataset_descriptor.test.feature_set = feature_set
        return model

    @classmethod
    def _bprint(cls, string):
        print('======================' + string.upper() + '======================')

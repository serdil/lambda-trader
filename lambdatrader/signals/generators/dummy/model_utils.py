from typing import List

from lambdatrader.signals.data_analysis.df_datasets import DateRange, SplitDateRange
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseAvgReturn, MaxReturn
from lambdatrader.signals.data_analysis.factories import SplitDateRanges
from lambdatrader.signals.generators.dummy.feature_spaces import all_sampler
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


    def apply_defaults(self):
        self.set_model_per_pair(True)
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

    def set_n_candles(self, n_candles):
        self.n_candles = n_candles
        self.close_value_set = DFFeatureSet(features=[CloseAvgReturn(n_candles=self.n_candles)])
        self.max_value_set = DFFeatureSet(features=[MaxReturn(n_candles=self.n_candles)])

    def set_model_per_pair(self, model_per_pair: bool):
        self.model_per_pair = model_per_pair
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

    def set_feature_set(self, feature_set):
        self.feature_set = feature_set

    def set_c_thr(self, c_thr):
        self.c_thr = c_thr

    def set_m_thr(self, m_thr):
        self.m_thr = m_thr

    def set_shrinking_feat_sel(self):
        self.select_features = True
        self.feat_sel_mode = self.FEAT_SEL_SHR

    def set_grow_shr_feat_sel(self):
        self.select_features = True
        self.feat_sel_mode = self.FEAT_SEL_GROW_SHR

    def set_hier_feat_sel(self):
        self.select_features = True
        self.feat_sel_mode = self.FEAT_SEL_HIER

    def set_feat_sel_n_target_feat(self, n_target_feat):
        self.n_target_feat = n_target_feat

    def set_feat_sel_sel_ratio(self, sel_ratio):
        self.sel_ratio = sel_ratio

    def set_feat_sel_n_rounds(self, feat_sel_n_rounds):
        self.feat_sel_n_rounds = feat_sel_n_rounds

    def use_xgb_model(self):
        self.model_to_use = self.MODEL_XGB

    def set_xgb_booster_params(self, booster_params):
        self.xgb_booster_params = booster_params

    def set_xgb_n_rounds(self, n_rounds):
        self.xgb_n_rounds = n_rounds

    def set_xgb_esr(self, early_stopping_rounds):
        self.xgb_esr = early_stopping_rounds

    def set_rdt_model(self):
        self.model_to_use = self.MODEL_RDT

    def execute(self):
        pass

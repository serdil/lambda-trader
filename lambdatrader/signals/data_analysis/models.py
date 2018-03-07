from operator import itemgetter

import xgboost as xgb
from xgboost.core import XGBoostError, DMatrix

from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor, XGBDMatrixDataset


class BaseModel:

    def train(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def predict_dataset_desc(self, desc):
        raise NotImplementedError

    def predict_ndarray(self, x):
        raise NotImplementedError

    def predict_df(self, df):
        raise NotImplementedError

    def predict_dmatrix(self, dmatrix):
        raise NotImplementedError

    @property
    def value_name(self):
        raise NotImplementedError


class NotTrainedException(Exception):
    pass


class XGBSplitDatasetModel(BaseModel):

    def __init__(self, dataset_descriptor: SplitDatasetDescriptor,
                 booster_params, num_round, early_stopping_rounds,
                 obj_name='', custom_obj=None,
                 use_saved_buffer=False, use_saved_libsvm_cache=False):
        self.dataset_descriptor = dataset_descriptor
        self.params = booster_params
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.obj_name = obj_name
        self.custom_obj = custom_obj

        if use_saved_buffer and use_saved_libsvm_cache:
            raise ValueError('use_saved_buffer and use_saved_libsvm_cache '
                             'can\'t be true at the same time')

        self.use_saved = use_saved_buffer or use_saved_libsvm_cache
        self.use_saved_buffer = use_saved_buffer
        self.use_saved_libsvm_cache = use_saved_libsvm_cache

        self.bst = None

    def train(self):
        dd = self.dataset_descriptor

        dtrain = XGBDMatrixDataset.compute(descriptor=dd.training, normalize=True,
                                           error_on_missing=False).dmatrix
        dval = XGBDMatrixDataset.compute(descriptor=dd.validation, normalize=True,
                                         error_on_missing=False).dmatrix
        dtest = XGBDMatrixDataset.compute(descriptor=dd.test, normalize=True,
                                          error_on_missing=False).dmatrix

        pred, real, bst = (self._train_xgb_with_dmatrices(dtrain, dval, dtest,
                                                          self.params,
                                                          self.num_round,
                                                          self.early_stopping_rounds,
                                                          self.obj_name))
        self.bst = bst
        return pred, real

    def _get_dmatrices(self):
        dd = self.dataset_descriptor

        if not self.use_saved:
            dtrain = XGBDMatrixDataset.compute(descriptor=dd.training, normalize=True,
                                               error_on_missing=False).dmatrix
            dval = XGBDMatrixDataset.compute(descriptor=dd.validation, normalize=True,
                                             error_on_missing=False).dmatrix
            dtest = XGBDMatrixDataset.compute(descriptor=dd.test, normalize=True,
                                              error_on_missing=False).dmatrix
        elif self.use_saved_buffer:
            dtrain = XGBDMatrixDataset.load_buffer(descriptor=dd.training).dmatrix
            dval = XGBDMatrixDataset.load_buffer(descriptor=dd.validation).dmatrix
            dtest = XGBDMatrixDataset.load_buffer(descriptor=dd.test).dmatrix
        else:
            dtrain = XGBDMatrixDataset.load_libsvm_cached(descriptor=dd.training).dmatrix
            dval = XGBDMatrixDataset.load_libsvm_cached(descriptor=dd.validation).dmatrix
            dtest = XGBDMatrixDataset.load_libsvm_cached(descriptor=dd.test).dmatrix
        return dtrain, dval, dtest

    def _train_xgb_with_dmatrices(self, dtrain, dval, dtest,
                                  params, num_round, early_stopping_rounds, obj_name):
        watchlist = [(dtrain, 'train_{}'.format(obj_name)), (dtest, 'test_{}'.format(obj_name)),
                     (dval, 'val_{}'.format(obj_name))]

        bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_round, evals=watchlist,
                        early_stopping_rounds=early_stopping_rounds, obj=self.custom_obj)

        feature_importances = bst.get_fscore()

        print()
        print('feature importances {}:'.format(obj_name))
        for f_name, imp in list(reversed(sorted(feature_importances.items(), key=itemgetter(1))))[
                           :10]:
            print(f_name, ':', imp)

        real = dtest.get_label()

        try:
            pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
        except XGBoostError:
            pred = bst.predict(dtest)
        return pred, real, bst

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def predict_dataset_desc(self, desc):
        dmatrix = XGBDMatrixDataset.compute(descriptor=desc, error_on_missing=True)
        return self.predict_dmatrix(dmatrix)

    def predict_ndarray(self, x):
        dmatrix = DMatrix(x, feature_names=self.dataset_descriptor.training.feature_names)
        return self.predict_dmatrix(dmatrix)

    def predict_df(self, df):
        dmatrix = DMatrix(df.values, feature_names=self.dataset_descriptor.training.feature_names)
        return self.predict_dmatrix(dmatrix)

    def predict_dmatrix(self, dmatrix):
        if self.bst is None:
            raise NotTrainedException
        try:
            return self.bst.predict(dmatrix, ntree_limit=self.bst.best_ntree_limit)
        except XGBoostError:
            return self.bst.predict(dmatrix)

    @property
    def value_name(self):
        return self.dataset_descriptor.first_value_name

from operator import itemgetter

import numpy as np
import xgboost as xgb
from math import sqrt

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.core import XGBoostError, DMatrix

from lambdatrader.signals.data_analysis.df_datasets import (
    SplitDatasetDescriptor, XGBDMatrixDataset, DFDataset,
)


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


class RFModel(BaseModel):

    def __init__(self,
                 dataset_descriptor: SplitDatasetDescriptor,
                 n_estimators=10, n_jobs=-1, max_depth=12, max_features='auto', min_samples_leaf=3,
                 obj_name=''):
        self.dataset_descriptor = dataset_descriptor

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf

        self.obj_name = obj_name

        self.forest = None

    def train(self):
        training_dd = self.dataset_descriptor.training
        x, y, feature_names = (DFDataset
                               .compute_from_descriptor(training_dd,
                                                        normalize=True,
                                                        error_on_missing=False)
                               .add_feature_values()
                               .add_value_values(value_name=self.value_name)
                               .add_feature_names()
                               .get())

        self.forest = RandomForestRegressor(n_estimators=self.n_estimators,
                                            n_jobs=self.n_jobs,
                                            max_depth=self.max_depth,
                                            max_features=self.max_features,
                                            min_samples_leaf=self.min_samples_leaf,
                                            verbose=True)

        self.forest.fit(x, y)

        importance = self.forest.feature_importances_
        name_importance = zip(feature_names, importance)
        name_importance_sorted = list(reversed(sorted(name_importance, key=itemgetter(1))))[:10]
        print()
        print('feature importances {}:'.format(self.obj_name))
        for f_name, imp in name_importance_sorted:
            print(f_name, ':', imp)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def predict_dataset_desc(self, desc):
        x = DFDataset.compute_from_descriptor(desc).add_feature_values().get()[0]
        return self.predict_ndarray(x)

    def predict_dmatrix(self, dmatrix):
        raise NotImplementedError

    def predict_df(self, df):
        return self.predict_ndarray(df.values)

    def predict_ndarray(self, x):
        if self.forest is None:
            raise NotTrainedException
        else:
            return self.forest.predict(x)

    @property
    def value_name(self):
        return self.dataset_descriptor.first_value_name


class BaggingDecisionTreeModel(BaseModel):

    def __init__(self,
                 dataset_descriptor: SplitDatasetDescriptor,
                 n_estimators=1000, max_samples=288, max_features='sqrt', dt_max_features='sqrt',
                 random_state=0, obj_name='', n_jobs=-1, oob_score=False):
        self.dataset_descriptor = dataset_descriptor

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.dt_max_features = dt_max_features

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score

        self.obj_name = obj_name

        self.forest = None

    def train(self):
        training_dd = self.dataset_descriptor.training
        x, y, feature_names = (DFDataset
                               .compute_from_descriptor(training_dd,
                                                        normalize=True,
                                                        error_on_missing=False)
                               .add_feature_values()
                               .add_value_values(value_name=self.value_name)
                               .add_feature_names()
                               .get())

        validation_dd = self.dataset_descriptor.validation
        val_x, val_y = (DFDataset
                        .compute_from_descriptor(validation_dd,
                                                 normalize=True,
                                                 error_on_missing=False)
                        .add_feature_values()
                        .add_value_values(value_name=self.value_name)
                        .get())
        print('train set size:', len(x))
        print('val set size:', len(val_x))

        dtr = DecisionTreeRegressor(max_features=self.dt_max_features,
                                    random_state=self.random_state)

        if self.max_features == 'sqrt':
            max_features = int(sqrt(len(feature_names)))
        else:
            max_features = self.max_features

        if self.max_samples == 'sqrt':
            max_samples = int(sqrt(len(x)))
        else:
            max_samples = self.max_samples

        self.forest = BaggingRegressor(base_estimator=dtr,
                                       n_estimators=self.n_estimators,
                                       max_samples=max_samples,
                                       max_features=max_features,
                                       n_jobs=self.n_jobs,
                                       random_state=self.random_state,
                                       oob_score=self.oob_score,
                                       verbose=True)

        self.forest.fit(x, y)
        try:
            val_pred = self.forest.predict(val_x)

            if self.oob_score:
                print()
                print('oob score {}: {}'.format(self.obj_name, self.forest.oob_score_))
                print()

            print()
            r2_score = metrics.r2_score(val_y, val_pred)
            rmse = sqrt(metrics.mean_squared_error(val_y, val_pred))
            print('r2 score:', r2_score)
            print('rmse:', rmse)
            print()

            self._print_feature_imp(self.forest, feature_names)
        except ValueError:
            print('ValueError during prediction for validation set.')

    @classmethod
    def _print_feature_imp(cls, forest, feature_names):
        feature_imp = cls._comp_feature_imp(forest, feature_names)
        name_importance_sorted = list(reversed(sorted(feature_imp, key=itemgetter(1))))[:20]
        print()
        print('feature importances:')
        for f_name, imp in name_importance_sorted:
            print(f_name, ':', imp)
        print()

    @classmethod
    def _comp_feature_imp(cls, forest, feature_names):
        feature_imp = np.zeros(len(feature_names))
        estimators = forest.estimators_
        estimators_features = forest.estimators_features_
        for est, est_features in zip(estimators, estimators_features):
            feature_imp[est_features] += est.feature_importances_
        feature_imp /= len(estimators)
        imp_tuples = list(zip(feature_names, feature_imp))
        return imp_tuples

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def predict_dataset_desc(self, desc):
        x = DFDataset.compute_from_descriptor(desc).add_feature_values().get()[0]
        return self.predict_ndarray(x)

    def predict_dmatrix(self, dmatrix):
        raise NotImplementedError

    def predict_df(self, df):
        return self.predict_ndarray(df.values)

    def predict_ndarray(self, x):
        if self.forest is None:
            raise NotTrainedException
        else:
            return self.forest.predict(x)

    @property
    def value_name(self):
        return self.dataset_descriptor.first_value_name

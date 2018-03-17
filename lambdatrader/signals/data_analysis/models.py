import logging
from math import sqrt
from operator import itemgetter

import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.core import XGBoostError, DMatrix

from lambdatrader.signals.data_analysis.df_datasets import (
    SplitDatasetDescriptor, XGBDMatrixDataset, DFDataset,
)
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.factories import FeatureSets


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

        # assigned during training
        self.bst = None
        self.feature_names = None
        self.feature_mapping = None
        self.reverse_feature_mapping = None
        self.feature_importance = None

    def train(self):
        dd = self.dataset_descriptor

        train_dmatrix_dataset = XGBDMatrixDataset.compute(descriptor=dd.training,
                                                          normalize=False,
                                                          error_on_missing=False)
        self.feature_names = train_dmatrix_dataset.feature_names
        self.feature_mapping = train_dmatrix_dataset.feature_mapping
        self.reverse_feature_mapping = train_dmatrix_dataset.reverse_feature_mapping

        dtrain = train_dmatrix_dataset.dmatrix
        dval = XGBDMatrixDataset.compute(descriptor=dd.validation, normalize=False,
                                         error_on_missing=False).dmatrix
        dtest = XGBDMatrixDataset.compute(descriptor=dd.test, normalize=False,
                                          error_on_missing=False).dmatrix

        pred, real, bst = (self._train_xgb_with_dmatrices(dtrain, dval, dtest,
                                                          self.params,
                                                          self.num_round,
                                                          self.early_stopping_rounds,
                                                          self.obj_name))
        self.bst = bst

        y_train_pred = self.bst.predict(dtrain)
        y_train_true = dtrain.get_label()

        y_val_pred = self.bst.predict(dval)
        y_val_true = dval.get_label()

        self._print_metrics('\ntraining metrics', y_train_true, y_train_pred, self.obj_name)
        self._print_metrics('\nvalidation metrics', y_val_true, y_val_pred, self.obj_name)

        self._comp_feature_imp_f_score()
        self._print_feature_imp()

        return pred, real

    def _get_dmatrices(self):
        dd = self.dataset_descriptor

        if not self.use_saved:
            dtrain = XGBDMatrixDataset.compute(descriptor=dd.training, normalize=False,
                                               error_on_missing=False).dmatrix
            dval = XGBDMatrixDataset.compute(descriptor=dd.validation, normalize=False,
                                             error_on_missing=False).dmatrix
            dtest = XGBDMatrixDataset.compute(descriptor=dd.test, normalize=False,
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

        real = dtest.get_label()

        try:
            pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
        except XGBoostError:
            pred = bst.predict(dtest)
        return pred, real, bst

    def _comp_feature_imp_f_score(self):
        fscore_dict = self.bst.get_fscore()
        self.feature_importance = list(reversed(sorted(fscore_dict.items(), key=itemgetter(1))))
        nonzero_set = set(fscore_dict.keys())
        zero_set = set(self.feature_names) - nonzero_set
        self.feature_importance.extend([(f_name, 0) for f_name in zero_set])

    def _comp_feature_imp_gain(self):
        pass

    def _comp_feature_imp_permutation(self):
        pass

    def _print_feature_imp(self):
        print()
        print('feature importances {}:'.format(self.obj_name))
        for f_name, imp in self.feature_importance[:10]:
            print(f_name, ':', imp)
        print()

    def select_features_by_ratio(self, ratio):
        num_select = int((len(self.feature_names) * ratio))
        return self.select_features_by_number(num_select)

    def select_features_by_number(self, n_select):
        num_discard = len(self.feature_names) - n_select
        f_name_ranks = [(f_name, rank) for rank, (f_name, _) in enumerate(self.feature_importance)]
        f_name_rank_mapping = dict(f_name_ranks)
        feature_lowest_ranks = []
        for feature, f_names in self.reverse_feature_mapping.items():
            lowest_rank = min([f_name_rank_mapping[f_name] for f_name in f_names])
            feature_lowest_ranks.append((feature, lowest_rank,))
        feature_lowest_ranks = list(reversed(sorted(feature_lowest_ranks, key=itemgetter(1))))
        num_discarded = 0
        prune_index = 0
        for i, (feature, lowest_rank) in enumerate(feature_lowest_ranks):
            # print('discarding {}, lowest rank: {}'.format(feature.name, lowest_rank))
            num_discarded += len(self.reverse_feature_mapping[feature])
            if num_discarded >= num_discard:
                print('highest discarded rank:', lowest_rank)
                prune_index = i
                break
        features = [feature for feature, _ in feature_lowest_ranks[prune_index + 1:]]
        return FeatureSets.compose_remove_duplicates(DFFeatureSet(features=features))

    @classmethod
    def _print_metrics(cls, message, y_true, y_pred, obj_name):
        print(message + ':')
        r2_score = metrics.r2_score(y_true, y_pred)
        rmse = sqrt(metrics.mean_squared_error(y_true, y_pred))
        print('r2 score {}:'.format(obj_name), r2_score)
        print('rmse {}:'.format(obj_name), rmse)
        print()

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
                 max_depth=12, random_state=0, obj_name='', n_jobs=-1, oob_score=False):
        self.dataset_descriptor = dataset_descriptor

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.dt_max_features = dt_max_features
        self.max_depth = max_depth

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score

        self.obj_name = obj_name

        # assigned during training
        self.forest = None
        self.feature_names = None
        self.feature_mapping = None
        self.reverse_feature_mapping = None
        self.feature_importance = None

    def train(self):
        training_dd = self.dataset_descriptor.training
        # TODO: normalize was made False
        (x, y,
         feature_names,
         feature_mapping,
         reverse_feature_mapping) = (DFDataset
                                     .compute_from_descriptor(training_dd,
                                                              normalize=False,
                                                              error_on_missing=False)
                                     .add_feature_values()
                                     .add_value_values(value_name=self.value_name)
                                     .add_feature_names()
                                     .add_feature_mapping()
                                     .add_reverse_feature_mapping()
                                     .get())

        self.feature_names = feature_names
        self.feature_mapping = feature_mapping
        self.reverse_feature_mapping = reverse_feature_mapping

        validation_dd = self.dataset_descriptor.validation
        val_x, val_y = (DFDataset
                        .compute_from_descriptor(validation_dd,
                                                 normalize=False,
                                                 error_on_missing=False)
                        .add_feature_values()
                        .add_value_values(value_name=self.value_name)
                        .get())
        print('train set size:', len(x))
        print('val set size:', len(val_x))

        if self.max_features == 'sqrt':
            max_features = int(sqrt(len(feature_names)))
        else:
            max_features = self.max_features

        if self.max_samples == 'sqrt':
            max_samples = int(sqrt(len(x)))
        else:
            max_samples = self.max_samples

        dtr = DecisionTreeRegressor(max_features=self.dt_max_features,
                                    random_state=self.random_state,
                                    max_depth=self.max_depth)

        self.forest = BaggingRegressor(base_estimator=dtr,
                                       n_estimators=self.n_estimators,
                                       max_samples=max_samples,
                                       max_features=max_features,
                                       n_jobs=self.n_jobs,
                                       random_state=self.random_state,
                                       oob_score=self.oob_score,
                                       verbose=True)

        self.forest.fit(x, y)

        if self.oob_score:
            print()
            print('oob score {}: {}'.format(self.obj_name, self.forest.oob_score_))
            print()

        try:
            train_pred = self.forest.predict(x)
            val_pred = self.forest.predict(val_x)

            self._print_metrics('\ntraining metrics', y, train_pred, self.obj_name)

            self._print_metrics('validation metrics', val_y, val_pred, self.obj_name)

            self._comp_feature_importance()
            self._print_feature_imp()
        except ValueError as e:
            print('ValueError during prediction for validation set.')
            logging.exception(e)

    @classmethod
    def _print_metrics(cls, message, y_true, y_pred, obj_name):
        print(message + ':')
        r2_score = metrics.r2_score(y_true, y_pred)
        rmse = sqrt(metrics.mean_squared_error(y_true, y_pred))
        print('r2 score {}:'.format(obj_name), r2_score)
        print('rmse {}:'.format(obj_name), rmse)
        print()

    def _comp_feature_importance(self):
        feature_imp = np.zeros(len(self.feature_names))
        estimators = self.forest.estimators_
        estimators_features = self.forest.estimators_features_
        for est, est_features in zip(estimators, estimators_features):
            feature_imp[est_features] += est.feature_importances_
        feature_imp /= len(estimators)
        imp_tuples = list(zip(self.feature_names, feature_imp))
        name_importance_sorted = list(reversed(sorted(imp_tuples, key=itemgetter(1))))
        self.feature_importance = name_importance_sorted

    def _print_feature_imp(self):
        print()
        print('feature importances {}:'.format(self.obj_name))
        for f_name, imp in self.feature_importance[:10]:
            print(f_name, ':', imp)
        print()

    def select_features_by_ratio(self, ratio):
        num_select = int((len(self.feature_names) * ratio))
        return self.select_features_by_number(num_select)

    def select_features_by_number(self, n_select):
        num_discard = len(self.feature_names) - n_select
        f_name_ranks = [(f_name, rank) for rank, (f_name, _) in enumerate(self.feature_importance)]
        f_name_rank_mapping = dict(f_name_ranks)
        feature_lowest_ranks = []
        for feature, f_names in self.reverse_feature_mapping.items():
            lowest_rank = min([f_name_rank_mapping[f_name] for f_name in f_names])
            feature_lowest_ranks.append((feature, lowest_rank,))
        feature_lowest_ranks = list(reversed(sorted(feature_lowest_ranks, key=itemgetter(1))))
        num_discarded = 0
        prune_index = 0
        for i, (feature, lowest_rank) in enumerate(feature_lowest_ranks):
            # print('discarding {}, lowest rank: {}'.format(feature.name, lowest_rank))
            num_discarded += len(self.reverse_feature_mapping[feature])
            if num_discarded >= num_discard:
                print('highest discarded rank:', lowest_rank)
                prune_index = i
                break
        features = [feature for feature, _ in feature_lowest_ranks[prune_index + 1:]]
        return FeatureSets.compose_remove_duplicates(DFFeatureSet(features=features))


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

    @property
    def feature_set(self):
        return self.dataset_descriptor.training.feature_set


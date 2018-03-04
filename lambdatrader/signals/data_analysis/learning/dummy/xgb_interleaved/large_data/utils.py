from operator import itemgetter

import xgboost as xgb
from xgboost.core import XGBoostError

from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor, XGBDMatrixDataset


def train_xgb(dataset_descriptor: SplitDatasetDescriptor,
              params, num_round, early_stopping_rounds, obj_name=''):
    dd = dataset_descriptor

    dtrain = XGBDMatrixDataset.compute(descriptor=dd.training,
                                       normalize=True, error_on_missing=False).dmatrix
    dval = XGBDMatrixDataset.compute(descriptor=dd.validation,
                                     normalize=True, error_on_missing=False).dmatrix
    dtest = XGBDMatrixDataset.compute(descriptor=dd.test,
                                      normalize=True, error_on_missing=False).dmatrix

    return train_xgb_with_dmatrices(dtrain, dval, dtest,
                                    params, num_round, early_stopping_rounds, obj_name)


def train_xgb_buffer(dataset_descriptor: SplitDatasetDescriptor,
                     params, num_round, early_stopping_rounds, obj_name=''):
    dd = dataset_descriptor

    dtrain = XGBDMatrixDataset.load_buffer(descriptor=dd.training).dmatrix
    dval = XGBDMatrixDataset.load_buffer(descriptor=dd.validation).dmatrix
    dtest = XGBDMatrixDataset.load_buffer(descriptor=dd.test).dmatrix

    return train_xgb_with_dmatrices(dtrain, dval, dtest, params, num_round, early_stopping_rounds,
                                    obj_name)


def train_xgb_libsvm_cache(dataset_descriptor: SplitDatasetDescriptor,
                           params, num_round, early_stopping_rounds, obj_name=''):
    dd = dataset_descriptor

    dtrain = XGBDMatrixDataset.load_libsvm_cached(descriptor=dd.training).dmatrix
    dval = XGBDMatrixDataset.load_libsvm_cached(descriptor=dd.validation).dmatrix
    dtest = XGBDMatrixDataset.load_libsvm_cached(descriptor=dd.test).dmatrix

    return train_xgb_with_dmatrices(dtrain, dval, dtest, params, num_round, early_stopping_rounds,
                                    obj_name)


def train_xgb_with_dmatrices(dtrain, dval, dtest,
                             params, num_round, early_stopping_rounds, obj_name=''):
    watchlist = [(dtrain, 'train_{}'.format(obj_name)), (dtest, 'test_{}'.format(obj_name)),
                 (dval, 'val_{}'.format(obj_name))]

    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_round,
                    evals=watchlist, early_stopping_rounds=early_stopping_rounds)

    close_best_ntree_limit = bst.best_ntree_limit
    feature_importances = bst.get_fscore()

    print()
    print('feature importances {}:'.format(obj_name))
    for f_name, imp in list(reversed(sorted(feature_importances.items(), key=itemgetter(1))))[
                       :10]:
        print(f_name, ':', imp)

    real = dtest.get_label()

    try:
        pred = bst.predict(dtest, ntree_limit=close_best_ntree_limit)
    except XGBoostError:
        pred = bst.predict(dtest)
    return pred, real, bst

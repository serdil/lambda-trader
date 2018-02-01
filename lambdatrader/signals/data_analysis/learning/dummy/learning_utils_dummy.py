import pprint
import time
from operator import itemgetter

import numpy as np
from sklearn import metrics


def train_and_test_model(dataset, model, classification_task=False, train_ratio=0.7):
    _feature_names = dataset.get_first_feature_names()

    X = dataset.get_numpy_feature_matrix()
    y = dataset.get_numpy_value_array()

    _n_samples = len(y)
    _split_ind = int(train_ratio * _n_samples)

    X_train = X[:_split_ind]
    y_train = y[:_split_ind]

    X_test = X[_split_ind:]
    y_test = y[_split_ind:]

    _start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - _start_time

    stats = {
        'training_time': training_time,
    }

    if hasattr(model, 'feature_importances_'):
        _importance = model.feature_importances_
        _name_importance = zip(_feature_names, _importance)
        name_importance_sorted = list(reversed(sorted(_name_importance, key=itemgetter(1))))[:20]
        stats['importance'] = name_importance_sorted

    test_stats = test_model(model, X_test, y_test, classification_task=classification_task)

    stats.update(test_stats)
    return stats


def test_model(model, x_test, y_test, classification_task=False):
    pred = model.predict(x_test)

    if not classification_task:
        mse = metrics.mean_squared_error(y_test, pred)
        score = model.score(x_test, y_test)

        pred_sign = pred > 0
        real_sign = y_test > 0

        sign_equal = np.equal(real_sign, pred_sign)

        n_samples = len(y_test)

        real_positive_ratio = np.sum(real_sign) / n_samples
        sign_equal_ratio = np.sum(sign_equal) / n_samples

        pred_real_sign = list(zip(pred_sign, real_sign))
        false_positive_count = sum([1 for elem in pred_real_sign if elem == (True, False)])
        true_positive_count = sum([1 for elem in pred_real_sign if elem == (True, True)])
        false_negative_count = sum([1 for elem in pred_real_sign if elem == (False, True)])
        true_negative_count = sum([1 for elem in pred_real_sign if elem == (False, False)])

        false_positive_ratio = false_positive_count / n_samples
        true_positive_ratio = true_positive_count / n_samples
        false_negative_ratio = false_negative_count / n_samples
        true_negative_ratio = true_negative_count / n_samples

        return {
            'pred': pred,
            'real': y_test,
            'pred_sign': pred_sign,
            'real_sign': real_sign,
            'sign_equal_ratio': sign_equal_ratio,
            'real_positive_ratio': real_positive_ratio,
            'mse': mse,
            'score': score,
            'false_positive_ratio': false_positive_ratio,
            'true_positive_ratio': true_positive_ratio,
            'false_negative_ratio': false_negative_ratio,
            'true_negative_ratio': true_negative_ratio,
        }


def print_model_stats(stats):
    if 'importance' in stats:
        print('IMPORTANCES:')
        for name, importance in stats['importance']:
            print('{:<80} {}'.format(name, importance))
        print()

    pred = stats['pred']
    real = stats['real']

    pred_real = list(zip(pred, real))
    print('pred, real:', pprint.pformat(pred_real))

    pred_sign = stats['pred_sign']
    real_sign = stats['real_sign']

    pred_real_sign = list(zip(pred_sign, real_sign))
    print('pred_sign, real_sign:', pprint.pformat(pred_real_sign))

    print()
    print('training time:', stats['training_time'])
    print('mse:', stats['mse'], 'score:', stats['score'])

    print()
    print('real positive ratio:', stats['real_positive_ratio'])
    print('sign equal ratio:', stats['sign_equal_ratio'])
    print()
    print('false positive ratio:', stats['false_positive_ratio'])
    print('true positive ratio:', stats['true_positive_ratio'])
    print()
    print('false negative ratio:', stats['false_negative_ratio'])
    print('true negative ratio:', stats['true_negative_ratio'])

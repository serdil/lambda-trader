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

        real_positive_ratio = np.sum(real_sign) / float(len(real_sign))
        sign_equal_ratio = np.sum(sign_equal) / float(len(sign_equal))


        return {
            'pred': pred,
            'real': y_test,
            'pred_sign': pred_sign,
            'real_sign': real_sign,
            'sign_equal_ratio': sign_equal_ratio,
            'real_positive_ratio': real_positive_ratio,
            'mse': mse,
            'score': score
        }


def print_model_stats(stats):
    if 'importance' in stats:
        print('IMPORTANCES:')
        for name, importance in stats['importance']:
            print('{:<80} {}'.format(name, importance))
        print()

    real = stats['real'] * 100
    preds = stats['pred'] * 100

    real_preds = list(zip(real, preds))
    print('real, pred:', pprint.pformat(real_preds))

    real_sign = stats['real_sign']
    pred_sign = stats['pred_sign']

    real_pred_sign = list(zip(real_sign, pred_sign))
    print('real_sign, pred_sign:', pprint.pformat(real_pred_sign))

    print()
    print('training time:', stats['training_time'])
    print('mse:', stats['mse'], 'score:', stats['score'])

    print()
    print('real positive ratio:', stats['real_positive_ratio'])
    print('sign equal ratio:', stats['sign_equal_ratio'])

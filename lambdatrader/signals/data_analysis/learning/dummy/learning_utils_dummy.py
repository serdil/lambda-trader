import pprint
import time
from collections import defaultdict
from operator import itemgetter

import numpy as np
from sklearn import metrics as sklearn_metrics


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

    metrics = {
        'training_time': training_time,
    }

    if not classification_task:
        if hasattr(model, 'feature_importances_'):
            _importance = model.feature_importances_
            _name_importance = zip(_feature_names, _importance)
            name_importance_sorted = list(reversed(sorted(_name_importance,
                                                          key=itemgetter(1))))[:20]
            metrics['importance'] = name_importance_sorted

    test_metrics = test_model(model, X_test, y_test, classification_task=classification_task)

    metrics.update(test_metrics)
    return metrics


def test_model(model, x_test, y_test, classification_task=False):
    pred = model.predict(x_test)
    real = y_test

    pred_real = list(zip(pred, real))
    score = model.score(x_test, y_test)

    if not classification_task:
        mse = sklearn_metrics.mean_squared_error(y_test, pred)

        pred_real_sorted = list(reversed(sorted(list(zip(pred, real)), key=itemgetter(0))))

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

        true_positive_ratio = true_positive_count / n_samples
        false_positive_ratio = false_positive_count / n_samples
        true_negative_ratio = true_negative_count / n_samples
        false_negative_ratio = false_negative_count / n_samples

        true_positive_total = sum([real for pred, real in pred_real_sorted
                                   if pred > 0 and real > 0])

        false_positive_total = sum([real for pred, real in pred_real_sorted
                                    if pred > 0 and real <= 0])

        if true_positive_count == 0:
            true_positive_count = 0.0001
        if false_positive_count == 0:
            false_positive_count = 0.0001

        true_positive_avg = true_positive_total / true_positive_count
        false_positive_avg = false_positive_total / false_positive_count

        positive_avg = (true_positive_total + false_positive_total) / \
                       (true_positive_count + false_positive_count)

        return {
            'pred': pred,
            'real': y_test,
            'pred_real_sorted': pred_real_sorted,
            'pred_sign': pred_sign,
            'real_sign': real_sign,
            'sign_equal_ratio': sign_equal_ratio,
            'real_positive_ratio': real_positive_ratio,
            'mse': mse,
            'score': score,
            'true_positive_ratio': true_positive_ratio,
            'false_positive_ratio': false_positive_ratio,
            'true_negative_ratio': true_negative_ratio,
            'false_negative_ratio': false_negative_ratio,
            'true_positive_total': true_positive_total,
            'false_positive_total': false_positive_total,
            'true_positive_avg': true_positive_avg,
            'false_positive_avg': false_positive_avg,
            'positive_avg': positive_avg
        }
    else:
        return {
            'pred': pred,
            'real': real,
            'pred_real': pred_real,
            'score': score
        }


def print_model_metrics(metrics_dict):
    metrics = defaultdict(str)
    metrics.update(metrics_dict)

    if 'importance' in metrics:
        print('IMPORTANCES:')
        for name, importance in metrics['importance']:
            print('{:<80} {}'.format(name, importance))
        print()

    pred_real = metrics['pred_real_sorted']
    print('pred, real:', pprint.pformat(pred_real))

    pred_sign = metrics['pred_sign']
    real_sign = metrics['real_sign']

    pred_real_sign = list(zip(pred_sign, real_sign))
    print('pred_sign, real_sign:', pprint.pformat(pred_real_sign))

    print()
    print('training time:', metrics['training_time'])
    print('mse:', metrics['mse'], 'score:', metrics['score'])

    print()
    print('real positive ratio:', metrics['real_positive_ratio'])
    print('sign equal ratio:', metrics['sign_equal_ratio'])
    print()
    print('true positive ratio:', metrics['true_positive_ratio'])
    print('false positive ratio:', metrics['false_positive_ratio'])
    print()
    print('true negative ratio:', metrics['true_negative_ratio'])
    print('false negative ratio:', metrics['false_negative_ratio'])
    print()
    print('true positive total:', metrics['true_positive_total'])
    print('false positive total:', metrics['false_positive_total'])
    print()
    print('true positive avg:', metrics['true_positive_avg'])
    print('false positive avg:', metrics['false_positive_avg'])
    print()
    print('positive avg:', metrics['positive_avg'])
    print()

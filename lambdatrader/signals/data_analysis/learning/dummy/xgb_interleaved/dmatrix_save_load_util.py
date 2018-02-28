import hashlib
import os

import xgboost as xgb

from lambdatrader.signals.data_analysis.learning.dummy.xgb_interleaved.utils import (
    get_close_dmatrix, get_max_dmatrix,
)
from lambdatrader.utilities.utils import get_project_directory


def _md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def _close_dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols,
                        valr, testr):
    n = _dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols, valr,
                      testr, 'close')
    n_t = '{}_{}.buffer'.format(n, 't')
    n_v = '{}_{}.buffer'.format(n, 'v')
    n_tt = '{}_{}.buffer'.format(n, 'tt')
    return n_t, n_v, n_tt


def _max_dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols, valr,
                      testr):
    n = _dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols, valr,
                      testr, 'max')
    n_t = '{}_{}.buffer'.format(n, 't')
    n_v = '{}_{}.buffer'.format(n, 'v')
    n_tt = '{}_{}.buffer'.format(n, 'tt')
    return n_t, n_v, n_tt


def _dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols, valr,
                  testr, _type):
    feature_names = ','.join([f.name for f in feature_set.features])
    symbols_names = ','.join(symbols)
    ident = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(_type, num_candles, candle_period, feature_names,
                                                num_days, days_offset, symbols_names, valr, testr)
    return str(_md5(ident))


def _dir_name():
    path = os.path.join(get_project_directory(), 'data', 'dmatrix')
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def _close_dmatrix_fpath(num_candles, candle_period, feature_set, num_days, days_offset, symbols,
                         valr, testr):
    t, v, tt = _close_dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset,
                                   symbols, valr, testr)
    return os.path.join(_dir_name(), t), os.path.join(_dir_name(), v), os.path.join(_dir_name(), tt)


def _max_dmatrix_fpath(num_candles, candle_period, feature_set, num_days, days_offset, symbols,
                       valr, testr):
    t, v, tt = _max_dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset,
                                 symbols, valr, testr)
    return os.path.join(_dir_name(), t), os.path.join(_dir_name(), v), os.path.join(_dir_name(), tt)


def load_close_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols,
                       valr, testr):
    t, v, tt = _close_dmatrix_fpath(num_candles, candle_period, feature_set, num_days, days_offset,
                                    symbols, valr, testr)
    return xgb.DMatrix(t), xgb.DMatrix(v), xgb.DMatrix(tt)


def save_close_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols,
                       valr, testr):
    dt, dv, dtt = get_close_dmatrix(symbols=symbols, feature_set=feature_set, val_ratio=valr / 10,
                                    test_ratio=testr / 10, day_offset=days_offset, days=num_days,
                                    num_candles=num_candles)
    nt, nv, ntt = _close_dmatrix_fpath(num_candles, candle_period, feature_set, num_days,
                                       days_offset, symbols, valr, testr)
    dt.save_binary(nt)
    dv.save_binary(nv)
    dtt.save_binary(ntt)


def load_max_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols, valr,
                     testr):
    t, v, tt = _close_dmatrix_fpath(num_candles, candle_period, feature_set, num_days, days_offset,
                                    symbols, valr, testr)
    return xgb.DMatrix(t), xgb.DMatrix(v), xgb.DMatrix(tt)


def save_max_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols, valr,
                     testr):
    dt, dv, dtt = get_max_dmatrix(symbols=symbols, feature_set=feature_set, val_ratio=valr / 10,
                                  test_ratio=testr / 10, day_offset=days_offset, days=num_days,
                                  num_candles=num_candles)
    nt, nv, ntt = _max_dmatrix_fpath(num_candles, candle_period, feature_set, num_days, days_offset,
                                     symbols, valr, testr)
    dt.save_binary(nt)
    dv.save_binary(nv)
    dtt.save_binary(ntt)

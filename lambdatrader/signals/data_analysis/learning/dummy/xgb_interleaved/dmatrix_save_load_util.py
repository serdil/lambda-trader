import hashlib


def md5(string):
    return hashlib.md5(string).hexdigest()


def close_dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols):
    return _dmatrix_name(num_candles, candle_period,
                         feature_set, num_days, days_offset, symbols, 'close')


def max_dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols):
    return _dmatrix_name(num_candles, candle_period,
                         feature_set, num_days, days_offset, symbols, 'max')


def _dmatrix_name(num_candles, candle_period, feature_set, num_days, days_offset, symbols, _type):
    feature_names = ','.join([f.name for f in feature_set.features])
    symbols_names = ','.join(symbols)
    ident = '{}_{}_{}_{}_{}_{}_{}'.format(_type, num_candles, candle_period,
                                          feature_names, num_days, days_offset, symbols_names)
    return md5(ident)


def get_close_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols):
    pass


def save_close_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols):
    pass


def get_max_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols):
    pass


def save_max_dmatrix(num_candles, candle_period, feature_set, num_days, days_offset, symbols):
    pass

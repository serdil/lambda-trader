import os
from datetime import datetime


def pair_from(first_currency, second_currency):
    return first_currency + '_' + second_currency


def pair_first(pair):
    return pair[:pair.index('_')]


def pair_second(pair):
    return pair[pair.index('_')+1:]


def get_now_timestamp():
    return datetime.utcnow().timestamp()


def timestamp_floor(timestamp):
    return timestamp - (timestamp % 300)


def timestamp_ceil(timestamp):
    return timestamp - (timestamp % 300) + 300


def get_one_day_seconds():
    return 24 * 3600


def get_n_day_seconds(n):
    return get_one_day_seconds() * n


def get_project_directory():
    return os.path.dirname(os.path.dirname(__file__))

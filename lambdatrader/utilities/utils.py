import os
from datetime import datetime

from lambdatrader.constants import M5


def pair_from(first_currency, second_currency):
    return first_currency + '_' + second_currency


def pair_first(pair):
    return pair[:pair.index('_')]


def pair_second(pair):
    return pair[pair.index('_')+1:]


def get_now_timestamp():
    return datetime.utcnow().timestamp()


def date_floor(date, period=M5):
    date = int(date)
    return date - (date % period.seconds())


def date_ceil(date, period=M5):
    date = int(date)
    return date - (date % period.seconds()) + period.seconds()


def get_one_day_seconds():
    return 24 * 3600


def get_n_day_seconds(n):
    return get_one_day_seconds() * n


def get_project_directory():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def running_in_docker():
    return os.path.isfile('/.dockerenv')

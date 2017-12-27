import os
from datetime import datetime

MINUTE_SECONDS = 60
HOUR_SECONDS = 60 * MINUTE_SECONDS
DAY_SECONDS = 24 * HOUR_SECONDS
WEEK_SECONDS = 7 * DAY_SECONDS
MONTH_SECONDS = 30 * DAY_SECONDS
YEAR_SECONDS = 365 * DAY_SECONDS



def pair_from(first_currency, second_currency):
    return first_currency + '_' + second_currency


def pair_first(pair):
    return pair[:pair.index('_')]


def pair_second(pair):
    return pair[pair.index('_')+1:]


def get_now_timestamp():
    return datetime.utcnow().timestamp()


def date_floor(date):
    date = int(date)
    return date - (date % 300)


def date_ceil(date):
    date = int(date)
    return date - (date % 300) + 300


def get_one_day_seconds():
    return 24 * 3600


def seconds(years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0):
    return seconds + minutes * MINUTE_SECONDS + hours * HOUR_SECONDS +\
           days * DAY_SECONDS + weeks * WEEK_SECONDS + months * MONTH_SECONDS + years * YEAR_SECONDS


def get_n_day_seconds(n):
    return get_one_day_seconds() * n


def get_project_directory():
    return os.path.dirname(os.path.dirname(__file__))


def running_in_docker():
    return os.path.isfile('/.dockerenv')

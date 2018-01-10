import os
import time

from lambdatrader.constants import M5

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
    return time.time()


def date_floor(date, period=M5):
    date = int(date)
    return date - (date % period.seconds())


def date_ceil(date, period=M5):
    date = int(date)
    return date - (date % period.seconds()) + period.seconds()


def get_one_day_seconds():
    return 24 * 3600


def seconds(years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0):
    return seconds + minutes * MINUTE_SECONDS + hours * HOUR_SECONDS +\
           days * DAY_SECONDS + weeks * WEEK_SECONDS + months * MONTH_SECONDS + years * YEAR_SECONDS


def candlesticks(years=0, months=0, weeks=0, days=0, hours=0, minutes=0, _seconds=0, period=M5):
    num_seconds = seconds(years=years, months=months, weeks=weeks,
                          days=days, hours=hours, minutes=minutes, seconds=_seconds)
    return int(num_seconds // period.seconds())


def get_n_day_seconds(n):
    return get_one_day_seconds() * n


def get_project_directory():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def running_in_docker():
    return os.path.isfile('/.dockerenv')

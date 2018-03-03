from datetime import datetime


def join_list(lst):
    return ','.join([str(elem) for elem in lst])


def date_str_to_timestamp(date_str):
    return int(datetime.strptime(date_str, '%Y-%m-%d').timestamp())

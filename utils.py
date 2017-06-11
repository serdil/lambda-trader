def pair_from(first_currency, second_currency):
    return first_currency + '_' + second_currency


def pair_first(pair):
    return pair[:pair.index('_')]


def pair_second(pair):
    return pair[pair.index('_')+1:]


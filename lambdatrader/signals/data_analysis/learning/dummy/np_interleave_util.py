from itertools import cycle, islice

import numpy as np


def round_robin(iterables):
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def interleave_2d(arrays, num_rows=288):
    max_len = int(max([len(array) for array in arrays]))
    n_split = int(max_len / num_rows)
    splitteds = [np.array_split(array, n_split) for array in arrays]
    stacked = np.vstack(round_robin(splitteds))
    return stacked


def interleave_1d(arrays, num_rows=288):
    max_len = int(max([len(array) for array in arrays]))
    n_split = int(max_len / num_rows)
    splitteds = [np.array_split(array, n_split) for array in arrays]
    stacked = np.hstack(round_robin(splitteds))
    return stacked

#
# a = np.array([1,3,5])
# b = np.array([2,4,6])
#
# print(interleave_1d([a, b], num_rows=1))
#
#
# a2d = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]).reshape(4, -1)
# b2d = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).reshape(4, -1)
#
#
# print()
# print(a2d)
# print()
# print(b2d)
# print()
# print(interleave_2d([a2d, b2d], num_rows=1))

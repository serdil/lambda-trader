import numpy as np


a_ones = np.ones(10, dtype=int)
a_range = np.arange(0, 10, 1)
a_random = np.random.randint(0, 10, 10)


print(a_ones)
print(a_range)
print(a_random)

print(np.logical_and(a_random > 5, a_random < 9))

# a_random[a_random > 5] = a_ones[a_random > 5]
a_random[a_random > 5] = -1

print(a_random)

print(np.full(5, np.nan))



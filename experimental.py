from numba import cuda
import numpy as np
import timeit
from numba import vectorize, float64
from math import sqrt


def mult(a, b):
    c = np.zeros(a.shape)
    for i in range(a.shape[0]):
        c[i] = sqrt(a[i]) * sqrt(b[i])

    return c


@vectorize([(float64(float64, float64))], target='cpu', cache=False)
def mult_cpu(a, b):
    return sqrt(a) * sqrt(b)


A = np.linspace(1, 1000000, 1000000)
B = np.linspace(1000001, 2000000, 1000000)


start = timeit.default_timer()
cx = mult_cpu(A, B)
print(f'time -> {(timeit.default_timer() - start)}')

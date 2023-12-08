from __future__ import print_function
from matrix_util import timer

import gevent

@timer
def naive_square_matrix_product(A, B):
    """ Implementation of naive squre matrix multiplication algorithm """
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


@timer
def parallel_square_matrix_product(A, B):
    """ Implementation of parallel version of
        naive squre matrix multiplication algorithm
    """
    def calc_ij(i, j, shape, A, B, C):
        """ evaluate C[i,j] element in the matrix """
        for k in range(shape):
            C[i][j] += A[i][k] * B[k][j]

    shape = len(A)
    # determine zero matrix
    C = [[0 for _ in range(shape)] for _ in range(shape)]
    # determine list of threads
    threads = []
    for i in range(shape):
        # parallel for i
        for j in range(shape):
            # parallel for j
            threads.append(gevent.spawn(calc_ij, i, j, shape, A, B, C))
    # join all threads
    gevent.joinall(threads)
    return C
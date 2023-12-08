from standard_multiplication import naive_square_matrix_product
from strassen_multiplication import strassen_square_matrix_product
from matrix_util import print_mx, timer
import numpy as np

time_naive = []
time_strassen = []
# for i in range(10):
input_shape = 100
a = np.random.random((input_shape, input_shape))
b = np.random.random((input_shape, input_shape))

print('naive algorithm')
naive = naive_square_matrix_product(a, b)
print('Strassen algorithm')
strassen = strassen_square_matrix_product(a, b)

from standard_multiplication import naive_square_matrix_product, parallel_square_matrix_product
from strassen_multiplication import strassen_square_matrix_product
from matrix_util import print_mx, timer, timer2
from alphatensor_multiplication import tensor_square_matrix_product
import numpy as np
import matplotlib.pyplot as plt
import time

time_naive_list = []
time_parellal_list = []
time_strassen_list = []
time_alpha_list = []
# for i in range(10):
inputs = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
for input_shape in inputs:
# input_shape = 4
    a = np.random.random((input_shape, input_shape))
    b = np.random.random((input_shape, input_shape))

    print('naive algorithm')
    start_time = time.time()
    naive = naive_square_matrix_product(a, b)
    time_naive = time.time() - start_time
    time_naive_list.append(time_naive)

    print('parallel naive product')
    start_time = time.time()
    parellal = parallel_square_matrix_product(a, b)
    time_parellal = time.time() - start_time
    time_parellal_list.append(time_naive)

    print('Strassen algorithm')
    start_time = time.time()
    strassen = strassen_square_matrix_product(a, b)
    time_strassen = time.time() - start_time
    time_strassen_list.append(time_strassen)

    print('tensor algorithm')
    start_time = time.time()
    alpha = tensor_square_matrix_product(a, b)
    time_alpha = time.time() - start_time
    time_alpha_list.append(time_alpha)


    print(alpha == strassen == naive)
    # print(time_naive, time_strassen, time_alpha)


plt.plot(inputs, time_naive_list)
plt.plot(inputs, time_strassen_list, '-')
plt.plot(inputs, time_alpha_list, '-.')
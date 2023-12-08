from standard_multiplication import naive_square_matrix_product 
from matrix_util import timer

def subtract(A, B):
    return [[x - y for x, y in zip(a, b)] for a, b in zip(A, B)]

def add(A, B):
    return [[x + y for x, y in zip(a, b)] for a, b in zip(A, B)]



# @timer
def tensor_square_matrix_product(A, B, leaf_size=16):
    """ Implementation of the strassen algorithm for square matrixes"""

    n = len(A)

    # leaf size determine
    # the size of matrix when we start using naive square matrix product
    if n <= leaf_size:
        return naive_square_matrix_product(A, B)

    # initializing the new sub-matrices
    new_size = n // 2

    a11 = list(map(lambda x: x[:new_size], A[:new_size]))      # top left
    a12 = list(map(lambda x: x[new_size:], A[:new_size]))      # top right
    a21 = list(map(lambda x: x[:new_size], A[new_size:]))      # bottom left
    a22 = list(map(lambda x: x[new_size:], A[new_size:]))      # bottom right

    b11 = list(map(lambda x: x[:new_size], B[:new_size]))      # top left
    b12 = list(map(lambda x: x[new_size:], B[:new_size]))      # top right
    b21 = list(map(lambda x: x[:new_size], B[new_size:]))      # bottom left
    b22 = list(map(lambda x: x[new_size:], B[new_size:]))      # bottom right

    # Calculating p1 to p7:
    # p1 = (a11 + a22) * (b11 + b22)
    p1 = tensor_square_matrix_product(add(a11, a22), add(b11, b22))
    # p2 = (a21 + a22) * (b11)
    p2 = tensor_square_matrix_product(add(a21, a22), b11)
    # p3 = (a11) * (b12 - b22)
    p3 = tensor_square_matrix_product(a11, subtract(b12, b22))
    # p4 = (a22) * (b21 - b11)
    p4 = tensor_square_matrix_product(a22, subtract(b21, b11))
    # p5 = (a11 + a22) * (b22)
    p5 = tensor_square_matrix_product(add(a11, a22), b22)
    # p6 = (a12 - a11) * (b11 + b12)
    p6 = tensor_square_matrix_product(subtract(a12, a11), add(b11, b12))
    # p7 = (a12 - a22) * (b21 + b22)
    p7 = tensor_square_matrix_product(subtract(a12, a22), add(b21, b22))

    # calculating c11 to c22:
    # c11 = p1 + p4 - p5+ p7
    c11 = add(subtract(add(p1, p4), p5), p7)
    # c12 = p1 + p2
    c12 = add(p3, p5)
    # c21 = p3 + p4
    c21 = add(p2, p4)
    # c22 = p1 - p2 + p3 + p6
    c22 = add(add(subtract(p1, p2), p3), p6)

    cl = c11 + c21
    cr = c12 + c22
    return [cl[i] + cr[i] for i in range(len(cl))]
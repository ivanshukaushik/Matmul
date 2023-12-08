from standard_multiplication import naive_square_matrix_product 
from matrix_util import timer

def subtract(A, B):
    return [[x - y for x, y in zip(a, b)] for a, b in zip(A, B)]

def add(A, B):
    return [[x + y for x, y in zip(a, b)] for a, b in zip(A, B)]

@timer
def strassen_square_matrix_product(A, B, leaf_size=64):
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
    # p1 = (a11) * (b12 - b22)
    p1 = strassen_square_matrix_product(a11, subtract(b12, b22))
    # p2 = (a11 + a12) * (b22)
    p2 = strassen_square_matrix_product(add(a11, a12), b22)
    # p3 = (a21 + a22) * (b11)
    p3 = strassen_square_matrix_product(add(a21, a22), b11)
    # p4 = (a22) * (b21 - b11)
    p4 = strassen_square_matrix_product(a22, subtract(b21, b11))
    # p5 = (a11 + a22) * (b11 + b22)
    p5 = strassen_square_matrix_product(add(a11, a22), add(b11, b22))
    # p6 = (a12 - a22) * (b21 + b22)
    p6 = strassen_square_matrix_product(subtract(a12, a22), add(b21, b22))
    # p7 = (a11 - a21) * (b11 + b12)
    p7 = strassen_square_matrix_product(subtract(a11, a21), add(b11, b12))

    # calculating c11 to c22:
    # c11 = p5 + p4 - p2 + p6
    c11 = add(subtract(add(p5, p4), p2), p6)
    # c12 = p1 + p2
    c12 = add(p1, p2)
    # c21 = p3 + p4
    c21 = add(p3, p4)
    # c22 = p5 + p1 - p3 - p7
    c22 = subtract(subtract(add(p5, p1), p3), p7)

    cl = c11 + c21
    cr = c12 + c22
    return [cl[i] + cr[i] for i in range(len(cl))]
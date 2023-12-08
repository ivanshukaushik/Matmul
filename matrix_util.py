import functools
import time

def print_mx(matrix):
    """ pretty print of matrix """
    for line in matrix:
        print("\t".join(map(str, line)))


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper

def timer2(func):
    def f(args):
        t1 = time.perf_counter()
        ret = func(args)
        t2 = time.perf_counter()
        print('execution time: {}'.format(t2-t1))
        return ret
    return f
import sys
import numpy as np
from ann import geneticnetwork
from ann.utils import connect_feedforward
import time

from rprop_time_test import get_test_data

def time_learn(rows, cols):
    '''Return time elapsed to learn 10000 iterations on data'''
    x, y = get_test_data(rows, cols)

    net = geneticnetwork(x.shape[1], 8, 1)
    connect_feedforward(net)

    net.generations=10

    # Time it
    start = time.time()
    net.learn(x, y)
    # Final time
    elapsed = time.time() - start

    return elapsed


if __name__ == '__main__':
    if len(sys.argv) < 3:
        rows, cols = 1000, 10
    else:
        rows, cols = sys.argv[1:3]
        rows, cols = int(rows), int(cols)

    elapsed = time_learn(rows, cols)
    print("Time to completion:", elapsed)

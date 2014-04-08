import sys
import numpy as np
from ann import rpropnetwork
from ann.utils import connect_feedforward
import time

def get_test_data(rows, columns):
    '''Hard coded to one output column.'''
    weights = np.random.normal(size=(columns+1))
    data = np.random.normal(size=(rows,columns+1))

    output = np.zeros((rows,1))
    output[:, 0] = np.sum(data * weights, axis=1)

    return (data[:, :-1], output)

def time_learn(rows, cols):
    '''Return time elapsed to learn 10000 iterations on data'''
    x, y = get_test_data(rows, cols)

    net = rpropnetwork(x.shape[1], 8, 1)
    connect_feedforward(net)

    net.maxError=0
    net.maxEpochs=1000

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

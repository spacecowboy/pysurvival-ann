"""
Functions related to the testing and validation of ANNs.
"""

import numpy as np
from numpy.random import shuffle
from . import get_C_index

def crossvalidate(ntimes, kfold, net, data, inputcols, targetcols):
    """
    Takes
    ntimes (repeat the cross validation this many times),
    kfold (split the data set this many pieces),
    a network (something with learn and output methods),
    data (2-dimensional numpy array),
    inputcols (what columns correspond to input data),
    targetcols (what columns correspond to target data),

    Returns a tuple (trnresults, valresults), where each
    piece is an array of results.
    """
    trnresults = []
    valresults = []

    # This might be a decimal number, remember to round it off
    indices = np.array(range(len(data)))
    kfrac = len(indices)/kfold

    for n in range(ntimes):
        # Re-shuffle the data every time
        shuffle(indices)

        # Divide all k parts among the indices.
        sets = [ indices[int(round(a*kfrac)) : int(round(b*kfrac))] for a, b in zip(range(kfold), range(1, kfold+1)) ]
        # Each time, either piece 0,1,...k will be the validation part
        for k in range(len(sets)):
            trnparts = list(range(len(sets)))
            # Remove validation set
            trnparts.remove(k)
            # Combine into actual sets
            trnindices = []
            for part in trnparts:
                trnindices.extend(sets[part])

            valindices = sets[k]

            # Train the network with previously set parameters on training set
            net.learn(data[trnindices][:, inputcols], data[trnindices][:, targetcols])

            # Training result
            predictions = np.array([net.output(x) for x in data[trnindices][:, inputcols]])
            c_index = get_C_index(data[trnindices][:, targetcols], predictions)

            trnresults.append(c_index)

            # Validation result
            predictions = np.array([net.output(x) for x in data[valindices][:, inputcols]])
            c_index = get_C_index(data[valindices][:, targetcols], predictions)

            valresults.append(c_index)

    return (trnresults, valresults)

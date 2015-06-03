#Expose functionality in c at package level
"""
A neural network package implemented in c++.

Should write intended usage here...
"""

from __future__ import division
from ._ann import (matrixnetwork as _matrixnetwork,
                   rpropnetwork as _rpropnetwork,
                   gennetwork as _gennetwork,
                   cindex, get_error,
                   get_deriv)
from ._ann import (ERROR_MSE, ERROR_SURV_MSE,
                   ERROR_SURV_LIKELIHOOD)
from .ensemble import Ensemble
from .utils import *
from functools import wraps
from random import uniform
import numpy as np


### Useful for multiple output neurons with softmax
def _softmax_predict(net, indata):
    '''
    Returns an integer, corresponding to the winning output neuron.
    Only valid if there are more than two output neurons and they use
    softmax activation functions.
    '''
    if net.output_count < 2:
        raise ValueError("Pointless to do class prediction with less than 2 outputs")
    if not np.all(net.activation_functions[-net.output_count:] == net.SOFTMAX):
        raise ValueError("Can only do softmax prediction if output is softmax.")

    return net.output(indata).argmax()


### Used in pickling

def _getmatrixstate(net):
    '''Returns the state of the network as a list of attributes:
    (neuron_numbers, attrs...)

    Where neuron_numbers = (numOfInputs, numOfHidden, numOfOutputs)'''
    neuron_numbers = (net.input_count, net.hidden_count,
                               net.output_count)
    attrs = {}
    attrs['neuron_numbers'] = neuron_numbers
    attrs['weights'] = net.weights
    attrs['activation_functions'] = list(net.activation_functions)
    attrs['connections'] = list(net.connections)

    return attrs


def _setmatrixstate(net, state):
    neurons = state.pop('neuron_numbers')
    net.__init__(*neurons)

    for k, v in state.items():
        setattr(net, k, v)
    pass


# Used for Matrix Networks
def _repr_matrix(net):

    return '''{}({}, {}, {})'''.format(net.__class__.__name__,
                                       net.input_count,
                                       net.hidden_count,
                                       net.output_count)

### These should be at the bottom of file ###


def UtilMatrix(cls):
    '''Adds util funcs to matrix networks'''
    cls.__repr__ = _repr_matrix
    cls.__getstate__ = _getmatrixstate
    cls.__setstate__ = _setmatrixstate
    cls.predict_class = _softmax_predict
    return cls


@UtilMatrix
class matrixnetwork(_matrixnetwork):
    __doc__ = _matrixnetwork.__doc__
    pass


@UtilMatrix
class rpropnetwork(_rpropnetwork):
    __doc__ = _rpropnetwork.__doc__
    pass


@UtilMatrix
class geneticnetwork(_gennetwork):
    __doc__ = _gennetwork.__doc__

    @wraps(_gennetwork.learn)
    def learn(self, trninputs, trnoutputs):
        # These fitness functions depend on sorted data
        if (self.fitness_function == _gennetwork.FITNESS_SURV_KAPLAN_MIN
          or self.fitness_function == _gennetwork.FITNESS_SURV_KAPLAN_MAX
          or self.fitness_function == _gennetwork.FITNESS_SURV_RISKGROUP_LOW
          or self.fitness_function == _gennetwork.FITNESS_SURV_RISKGROUP_HIGH
          or self.fitness_function == _gennetwork.FITNESS_TARONEWARE_MEAN
          or self.fitness_function == _gennetwork.FITNESS_TARONEWARE_HIGHLOW):
            asc = trnoutputs[:, 0].argsort()
            super(geneticnetwork, self).learn(trninputs[asc], trnoutputs[asc])
        else:
            # Pass data through unchanged
            super(geneticnetwork, self).learn(trninputs, trnoutputs)


@UtilMatrix
class gensurvnetwork(_gennetwork):
    __doc__ = _gennetwork.__doc__

    def riskeval(self, inputdata):
        '''Given the inputs for a patient, returns its normalized
        relative rank, compared to the training data.

        Input: An array of equal length as numOfInputs
        Returns: A value between 0 and 1
        '''
        if not hasattr(self, 'trnoutputs'):
            raise ValueError('You have to train the network first')

        out = self.output(inputdata)
        # How many have better expected survival
        idx = len(self.trnoutputs[self.trnoutputs > out])
        return idx / len(self.trnoutputs)

    @wraps(_gennetwork.learn)
    def learn(self, trninputs, trntargets):
        super(gensurvnetwork, self).learn(trninputs, trntargets)
        # Evaluate and store outputs for training data
        self.trnoutputs = np.array([self.output(x) for x in trninputs]).ravel()

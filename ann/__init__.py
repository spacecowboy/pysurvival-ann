#Expose functionality in c at package level
"""
A neural network package implemented in c++.

Should write intended usage here...
"""

from ._ann import (ffnetwork, rpropnetwork,
                   gensurvnetwork as _gensurvnetwork, get_C_index)
from .ensemble import Ensemble
from random import uniform
import numpy as np

class gensurvnetwork(_gensurvnetwork):
    '''Genetic survival network.'''

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

    def learn(self, trninputs, trntargets):
        '''Trains the network using a genetic algorithm'''
        super(gensurvnetwork, self).learn(trninputs, trntargets)
        # Evaluate and store outputs for training data
        self.trnoutputs = np.array([self.output(x) for x in trninputs]).ravel()

def getSingleLayerRProp(numOfInputs, numOfHidden, numOfOutputs):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with rprop.'''
    net = rpropnetwork(numOfInputs, numOfHidden, numOfOutputs)

    return connectAsSingleLayer(net)

def getSingleLayerGenSurv(numOfInputs, numOfHidden):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with a genetic algorithm
    for censored survival data.

    ORDER MATTERS!
    '''
    net = gensurvnetwork(numOfInputs, numOfHidden)

    for h in range(net.numOfHidden):
        net.connectHToB(h, uniform(-0.5, 0.5))
        net.connectOToH(0, h, uniform(-0.5, 0.5))
        for i in range(net.numOfInputs):
            net.connectHToI(h, i, uniform(-0.5, 0.5))

    net.connectOToB(0, uniform(-0.5, 0.5))

    net.setOutputFunction(net.LINEAR)

    return net


def connectAsSingleLayer(net):
    '''Given an unconnected network, will connect it as a single layer'''
    net.setOutputFunction(net.LOGSIG)
    net.setHiddenFunction(net.TANH)

    for h in range(net.numOfHidden):
        net.connectHToB(h, uniform(-0.5, 0.5))
        for i in range(net.numOfInputs):
            net.connectHToI(h, i, uniform(-0.5, 0.5))

    for o in range(net.numOfOutputs):
        net.connectOToB(o, uniform(-0.5, 0.5))
        for h in range(net.numOfHidden):
            net.connectOToH(o, h, uniform(-0.5, 0.5))

    return net

#Expose functionality in c at package level
"""
A neural network package implemented in c++.

Should write intended usage here...
"""

from __future__ import division
from ._ann import (ffnetwork as _ffnetwork, rpropnetwork as _rpropnetwork,
                   gensurvnetwork as _gensurvnetwork, get_C_index)
from .ensemble import Ensemble
from random import uniform
import numpy as np


def getSingleLayerRProp(numOfInputs, numOfHidden, numOfOutputs):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with rprop.'''
    net = rpropnetwork(numOfInputs, numOfHidden, numOfOutputs)

    connectAsSingleLayer(net)
    return net

def getRProp(numOfInputs, hiddenlayers, numOfOutputs):
    ''' Constructs and connects a neural network with n layers
    of hidden neurons which can be trained with rprop.'''
    numOfHidden = sum(hiddenlayers)
    net = rpropnetwork(numOfInputs, numOfHidden, numOfOutputs)

    connectAsNLayer(net, hiddenlayers)
    return net

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

    net.hiddenActivationFunction = net.TANH
    net.outputActivationFunction = net.LINEAR

    return net


def connectAsSingleLayer(net):
    '''Given an unconnected network, will connect it as a single layer'''
    net.outputActivationFunction = net.LOGSIG
    net.hiddenActivationFunction = net.TANH

    for h in range(net.numOfHidden):
        net.connectHToB(h, uniform(-0.5, 0.5))
        for i in range(net.numOfInputs):
            net.connectHToI(h, i, uniform(-0.5, 0.5))

    for o in range(net.numOfOutputs):
        net.connectOToB(o, uniform(-0.5, 0.5))
        for h in range(net.numOfHidden):
            net.connectOToH(o, h, uniform(-0.5, 0.5))

def connectAsNLayer(net, layers):
    '''Given an unconnected network, will connect it as layers
    where second argument describes how to place the hidden nodes
    in layers. Example: [10, 3]'''
    net.outputActivationFunction = net.LOGSIG
    net.hiddenActivationFunction = net.TANH

    if sum(layers) != net.numOfHidden:
        raise ValueError("Sum of layers do not match numOfHidden!")

    prev_layer = []
    total_count = 0
    for layer_count in layers:
        current_layer = range(total_count, total_count + layer_count)
        for h in current_layer:
            net.connectHToB(h, uniform(-0.5, 0.5))
            # Input for first layer
            if len(prev_layer) == 0:
                for i in range(net.numOfInputs):
                    net.connectHToI(h, i, uniform(-0.5, 0.5))
            else:
                for i in prev_layer:
                    net.connectHToH(h, i, uniform(-0.5, 0.5))

        total_count += layer_count
        prev_layer = current_layer

    for o in range(net.numOfOutputs):
        net.connectOToB(o, uniform(-0.5, 0.5))
        for h in prev_layer:
            net.connectOToH(o, h, uniform(-0.5, 0.5))


def getWeights(net):
    '''Returns a 2d-array of the weights in this network.
    Ordered as rows x cols: (hidden-output)x(bias-input-hidden-output)'''

    numOfCols = 1 + net.numOfInputs + net.numOfHidden + net.numOfOutputs

    rows = None
    #Hidden neurons
    for i in range(0, net.numOfHidden):
        row = np.empty(numOfCols)
        row[:] = None

        inputWeights = net.getInputWeightsOfHidden(i)
        for id, weight in inputWeights.items():
            row[1 + id] = weight

        neuronWeights = net.getNeuronWeightsOfHidden(i)
        for id, weight in neuronWeights.items():
            if (id == -1):
                #Bias
                row[0] = weight
            else:
                #Hidden neuron
                row[1 + numOfInputs + id] = weight

        if rows is None:
            #This is total array so far
            rows = np.row_stack((row,))
        else:
            # Append to total array
            rows = np.row_stack((rows, row))

    #Output neurons
    for i in range(0, net.numOfOutputs):
        row = np.empty(numOfCols)
        row[:] = None

        inputWeights = net.getInputWeightsOfOutput(i)
        for id, weight in inputWeights.items():
            row[1 + id] = weight

        neuronWeights = net.getNeuronWeightsOfOutput(i)
        for id, weight in neuronWeights.items():
            if (id == -1):
                #Bias
                row[0] = weight
            else:
                #Hidden neuron
                row[1 + net.numOfInputs + id] = weight

        if rows is None:
            #This is total array so far
            rows = np.row_stack((row,))
        else:
            # Append to total array
            rows = np.row_stack((rows, row))

    return rows

def _getstate(net):
    '''Returns the state of the network as a tuple:
    (neuron_numbers, hiddenActivationFunction, outputActivationFunction, weights)

    Where neuron_numbers = (numOfInputs, numOfHidden, numOfOutputs)
    and weights = net.getWeights()'''
    neuron_numbers = (net.numOfInputs, net.numOfHidden,
                      net.numOfOutputs)
    return (neuron_numbers,
            net.hiddenActivationFunction,
            net.outputActivationFunction,
            net.getWeights())

def _setstate(net, state):
    '''Restores the state of the network from the given state.

    See __getstate() for info on format.'''
    net.__init__(*state[0])

    net.hiddenActivationFunction = state[1]
    net.outputActivationFunction = state[2]

    weights = state[3]
    # Connect hidden neurons
    for id, neuronWeights in enumerate(weights[:net.numOfHidden]):
        # bias
        if not np.isnan(neuronWeights[0]):
            net.connectHToB(id, neuronWeights[0])
        # inputs
        left  = 1
        right = left + net.numOfInputs
        for idx, weight in enumerate(neuronWeights[left:right]):
            if not np.isnan(weight):
                net.connectHToI(id, idx, weight)
        # hidden
        left = right
        right = left + net.numOfHidden
        for targetId, weight in enumerate(neuronWeights[left:right]):
            if not np.isnan(weight):
                net.connectHToH(id, targetId, weight)

    # Connect output neurons
    for id, neuronWeights in enumerate(weights[net.numOfHidden:]):
        # bias
        if not np.isnan(neuronWeights[0]):
            net.connectOToB(id, neuronWeights[0])
        # inputs
        left  = 1
        right = left + net.numOfInputs
        for idx, weight in enumerate(neuronWeights[left:right]):
            if not np.isnan(weight):
                net.connectOToI(id, idx, weight)
        # hidden
        left = right
        right = left + net.numOfHidden
        for targetId, weight in enumerate(neuronWeights[left:right]):
            if not np.isnan(weight):
                net.connectOToH(id, targetId, weight)


def _repr(net):
    funcVals = {net.LINEAR: 'LINEAR',
                net.TANH: 'TANH',
                net.LOGSIG: 'LOGSIG'}

    return """{}({}, {}, {})
Hidden: {}
Output: {}""".format(net.__class__.__name__,
    net.numOfInputs, net.numOfHidden, net.numOfOutputs,
    funcVals.get(net.hiddenActivationFunction),
    funcVals.get(net.outputActivationFunction))

### These should be at the bottom of file ###

def UtilFuncs(cls):
    '''Adds the common utility functions to neural networks.'''
    cls.getWeights = getWeights
    cls.connectAsNLayer = connectAsNLayer
    cls.connectAsSingleLayer = connectAsSingleLayer
    cls.__getstate__ = _getstate
    cls.__setstate__ = _setstate
    cls.__repr__ = _repr
    return cls

@UtilFuncs
class ffnetwork(_ffnetwork):
    pass

@UtilFuncs
class rpropnetwork(_rpropnetwork):
    pass

@UtilFuncs
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

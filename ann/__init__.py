#Expose functionality in c at package level
"""
A neural network package implemented in c++.

Should write intended usage here...
"""

from __future__ import division
from ._ann import (ffnetwork as _ffnetwork, rpropnetwork as _rpropnetwork,
                   gennetwork as _gennetwork,
                   gensurvnetwork as _gensurvnetwork, get_C_index,
                   cascadenetwork as _cascadenetwork,
                   coxcascadenetwork as _coxcascadenetwork,
                   geneticcascadenetwork as _geneticcascadenetwork,
                   geneticladdernetwork as _geneticladdernetwork)
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

def getSingleLayerGenSurv(numOfInputs, numOfHidden, variance=0.5):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with a genetic algorithm
    for censored survival data. Weights are scaled so that each
    vector of weights connected to a neuron will have L2-norm = 1.

    Keyword arguments:
    numOfInputs - Number of input neurons
    numOfHidden - Number of hidden neurons
    variance=0.5 - Uniform variance of initial weights

    ORDER MATTERS!
    '''
    net = gensurvnetwork(numOfInputs, numOfHidden)

    outputweights = np.random.normal(size = numOfHidden + 1)
    outputweights /= np.linalg.norm(outputweights)

    for h in range(net.numOfHidden):
        hiddenweights = np.random.normal(size = numOfInputs + 1)
        hiddenweights /= np.linalg.norm(hiddenweights)

        net.connectHToB(h, hiddenweights[-1])
        net.connectOToH(0, h, outputweights[h])
        for i in range(net.numOfInputs):
            net.connectHToI(h, i, hiddenweights[i])

    net.connectOToB(0, outputweights[-1])

    net.hiddenActivationFunction = net.LOGSIG
    net.outputActivationFunction = net.LOGSIG

    return net

def getSingleLayerGenetic(numOfInputs, numOfHidden, numOfOutputs):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with a genetic algorithm
    for censored survival data. Weights are scaled so that each
    vector of weights connected to a neuron will have L2-norm = 1.

    Keyword arguments:
    numOfInputs - Number of input neurons
    numOfHidden - Number of hidden neurons
    numOfOutputs - Number of output neurons

    ORDER MATTERS!
    '''
    net = geneticnetwork(numOfInputs, numOfHidden, numOfOutputs)

    outputweights = np.random.normal(size = numOfHidden + 1)
    outputweights /= np.linalg.norm(outputweights)

    for h in range(net.numOfHidden):
        hiddenweights = np.random.normal(size = numOfInputs + 1)
        hiddenweights /= np.linalg.norm(hiddenweights)

        net.connectHToB(h, hiddenweights[-1])
        net.connectOToH(0, h, outputweights[h])
        for i in range(net.numOfInputs):
            net.connectHToI(h, i, hiddenweights[i])

    net.connectOToB(0, outputweights[-1])

    net.hiddenActivationFunction = net.LOGSIG
    net.outputActivationFunction = net.LOGSIG

    return net


def getCascadeNetwork(numOfInputs):
    '''Returns a connected cascade network with the specified
    amount of input neurons, ready to be trained.
    '''
    net = cascadenetwork(numOfInputs)
    connectAsShortcutNLayer(net, [])
    return net

def getCoxCascadeNetwork(numOfInputs):
    '''Returns a connected cox cascade network with the specified
    amount of input neurons, ready to be trained.
    '''
    net = coxcascadenetwork(numOfInputs)
    connectAsShortcutNLayer(net, [])
    return net


def getGeneticCascadeNetwork(numOfInputs):
    '''Returns a connected genetic cascade network with the specified
    amount of input neurons, ready to be trained.
    '''
    net = geneticcascadenetwork(numOfInputs)
    net.hiddenActivationFunction = net.LOGSIG
    net.outputActivationFunction = net.LOGSIG
    connectAsShortcutNLayer(net, [])
    return net

def getGeneticLadderNetwork(numOfInputs):
    '''Returns a connected genetic ladder network with the specified
    amount of input neurons, ready to be trained.
    '''
    net = geneticladdernetwork(numOfInputs)
    connectAsShortcutNLayer(net, [])
    net.hiddenActivationFunction = net.LOGSIG
    net.outputActivationFunction = net.LOGSIG
    return net


def connectAsSingleLayer(net):
    '''Given an unconnected network, will connect it as a single layer.
    Also scales the weight vectors appropriately.'''
    net.outputActivationFunction = net.LOGSIG
    net.hiddenActivationFunction = net.LOGSIG

    outputweights = np.random.normal(size = numOfHidden + 1)
    outputweights /= np.linalg.norm(outputweights)

    for h in range(net.numOfHidden):
        hiddenweights = np.random.normal(size = numOfInputs + 1)
        hiddenweights /= np.linalg.norm(hiddenweights)

        net.connectHToB(h, hiddenweights[-1])
        net.connectOToH(0, h, outputweights[h])
        for i in range(net.numOfInputs):
            net.connectHToI(h, i, hiddenweights[i])

    net.connectOToB(0, outputweights[-1])

def connectAsNLayer(net, layers):
    '''Given an unconnected network, will connect it as layers
    where second argument describes how to place the hidden nodes
    in layers. Example: [10, 3]'''
    net.outputActivationFunction = net.LOGSIG
    net.hiddenActivationFunction = net.LOGSIG

    if sum(layers) != net.numOfHidden:
        raise ValueError("Sum of layers do not match numOfHidden!")

    prev_layer = []
    total_count = 0
    for layer_count in layers:
        current_layer = range(total_count, total_count + layer_count)
        for h in current_layer:
            # Input for first layer
            if len(prev_layer) == 0:
                weights = np.random.normal(size = net.numOfInputs + 1)
                weights /= np.linalg.norm(weights)
                for i in range(net.numOfInputs):
                    net.connectHToI(h, i, weights[i])
            else:
                weights = np.random.normal(size = len(prev_layer) + 1)
                weights /= np.linalg.norm(weights)
                for j, i in enumerate(prev_layer):
                    net.connectHToH(h, i, weights[j])
            # Bias
            net.connectHToB(h, weights[-1])

        total_count += layer_count
        prev_layer = current_layer

    for o in range(net.numOfOutputs):
        weights = np.random.normal(size = len(prev_layer) + 1)
        weights /= np.linalg.norm(weights)
        net.connectOToB(o, weights[-1])
        for i, h in enumerate(prev_layer):
            net.connectOToH(o, h, weights[i])

def connectAsShortcutNLayer(net, layers):
    '''
    All neurons are connected to all previous neurons/inputs.
    Second argument describes hidden layer structure. Each layer
    is fully connected to all previous layers.

    E.g. layers = [4, 2].
    '''
    net.outputActivationFunction = net.LOGSIG
    net.hiddenActivationFunction = net.TANH

    if sum(layers) != net.numOfHidden:
        raise ValueError("Sum of layers do not match numOfHidden!")

    prev_layer = []
    prev_layers = []
    total_count = 0
    for l, layer_count in enumerate(layers):
        current_layer = range(total_count, total_count + layer_count)
        for h in current_layer:
            net.connectHToB(h, uniform(-0.5, 0.5))

            for i in range(net.numOfInputs):
                net.connectHToI(h, i, uniform(-0.5, 0.5))

            # Only previous layers
            for i in prev_layer[:i]:
                net.connectHToH(h, i, uniform(-0.5, 0.5))

        total_count += layer_count
        prev_layer = current_layer
        prev_layers.append(prev_layer)

    for o in range(net.numOfOutputs):
        net.connectOToB(o, uniform(-0.5, 0.5))
        for i in range(net.numOfInputs):
            net.connectOToI(o, i, uniform(-0.5, 0.5))

        for neurons in prev_layers:
            for h in neurons:
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
                row[1 + net.numOfInputs + id] = weight

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
class cascadenetwork(_cascadenetwork):
    pass

@UtilFuncs
class coxcascadenetwork(_coxcascadenetwork):
    pass

@UtilFuncs
class geneticcascadenetwork(_geneticcascadenetwork):
    pass

@UtilFuncs
class geneticladdernetwork(_geneticladdernetwork):
    pass

@UtilFuncs
class geneticnetwork(_gennetwork):
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

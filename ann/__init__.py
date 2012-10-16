#Expose functionality in c at package level
"""
A neural network package implemented in c++.

Should write intended usage here...
"""

from ._ann import *
from random import uniform

def getSingleLayerRProp(numOfInputs, numOfHidden, numOfOutputs):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with rprop.'''
    net = rpropnetwork(numOfInputs, numOfHidden, numOfOutputs)

    return connectAsSingleLayer(net)

def getSingleLayerGenSurv(numOfInputs, numOfHidden):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with a genetic algorithm
    for censored survival data.'''
    net = gensurvnetwork(numOfInputs, numOfHidden)

    return connectAsSingleLayer(net)

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

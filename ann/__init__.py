#Expose functionality in c at package level
from _ann import *
from random import uniform

def getSingleLayerRProp(numOfInputs, numOfHidden, numOfOutputs):
    ''' Constructs and connects a neural network with a single layer
    of hidden neurons which can be trained with rprop.'''
    net = rpropnetwork(numOfInputs, numOfHidden, numOfOutputs)

    net.setOutputFunction(net.LOGSIG)
    net.setHiddenFunction(net.TANH)

    for h in range(numOfHidden):
        net.connectHToB(h, uniform(-0.5, 0.5))
        for i in range(numOfInputs):
            net.connectHToI(h, i, uniform(-0.5, 0.5))


    for o in range(numOfOutputs):
        net.connectOToB(o, uniform(-0.5, 0.5))
        for h in range(numOfHidden):
            net.connectOToH(o, h, uniform(-0.5, 0.5))


    return net

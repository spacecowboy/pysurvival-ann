"""Some utility functions for dealing with ANNs."""

from __future__ import division
import numpy as np


def connect_feedforward(net, layers=None, hidden_act=None, out_act=None):
    '''Connect a neural network in a feedforward manner.

    Parameters:
    net - The network to connect.
    layers - A list of hidden layer sizes (ints). The first hidden layer is
             connected to the inputs. The last is connected to the outputs.
             Must sum to number of hidden neurons! If None, assumes a
             single hidden layer is desired.
    hidden_act - Activation function of hidden layers, default logsig
    out_act - Activation function of output layer, default logsig

    Example:
    >>> net = rpropnetwork(2, 10, 1)
    >>> connect_feedforward(net)

    Which is equivalent to:
    >>> connect_feedforward(net, [10])

    Additional layers can be specified as follows:
    >>> connect_feedforward(net, [5, 5])
    >>> connect_feedforward(net, [2,3,3,2])
    '''
    if layers is None:
        layers = [net.hidden_count]

    if sum(layers) != net.hidden_count:
        raise ValueError("Sum of layers must equal number of hidden neurons")

    # plus one for bias
    dim = net.input_count + net.hidden_count + net.output_count + 1
    conns = net.connections.reshape((dim, dim))
    weights = net.weights.reshape((dim, dim))
    # reset connections
    conns[:, :] = 0
    weights[:, :] = 0

    # connect all to bias
    conns[:, net.input_count] = 1
    weights[:, net.input_count] = np.random.normal(size=dim)

    hidden_start = net.input_count + 1

    # connect first hidden layer to inputs
    conns[hidden_start:hidden_start + layers[0],
          0:net.input_count] = 1
    weights[hidden_start:hidden_start + layers[0],
            0:net.input_count] = np.random.normal(size=(layers[0],
                                                        net.input_count))

    prev_start = hidden_start
    prev_end = hidden_start + layers[0]
    # connect hidden layers to each other
    for layer in layers[1:]:
        conns[prev_end:prev_end + layer,
              prev_start:prev_end] = 1
        weights[prev_end:prev_end + layer,
                prev_start:prev_end] = np.random.normal(
                    size=(layer, prev_end - prev_start))
        prev_start = prev_end
        prev_end += layer

    output_start = 1 + net.input_count + net.hidden_count
    # connect outputs to last hidden layer
    conns[output_start:,
          output_start - layers[-1]:output_start] = 1
    weights[output_start:,
            output_start - layers[-1]:output_start] = np.random.normal(
                size=(net.output_count, layers[-1]))

    net.connections = conns.reshape(dim*dim)
    net.weights = weights.reshape(dim*dim)

    # set activation functions
    act = net.activation_functions

    ha = net.LOGSIG
    if hidden_act is not None:
        ha = hidden_act

    oa = net.LOGSIG
    if out_act is not None:
        oa = out_act

    act[hidden_start:] = ha
    act[output_start:] = oa

    net.activation_functions = act

#!/usr/bin/env python
import numpy as np

def getXOR():
    '''Returns a tuple of (inputs, targets)'''
    inputs = np.array([[0.0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    outputs = np.array([[0.0],
                        [1],
                        [1],
                        [0]])
    return (inputs, outputs)

def test_import():
    import ann

def test_matrixnetwork():
    from ann import matrixnetwork

    net = matrixnetwork(2, 2, 1)

    # Default start is zero
    xor_in, xor_out = getXOR()
    for val in xor_in:
        assert net.output(val) == 0, "Expected zero output as default"

    length = net.input_count + net.hidden_count + net.output_count + 1

    # Weights
    print("Length: ", len(net.weights))
    print(net.weights)
    assert len(net.weights) == length**2, "Wrong length of weight vector"
    assert np.all(net.weights == 0) == True, "Expected weights to equal zero"

    weights = net.weights

    for i in range(len(weights)):
        weights[i] = 1.0
    net.weights = weights

    print(net.weights)
    assert np.all(net.weights == 1.0) == True, "Expected weights to equal 1"

    # Connections
    print("Length: ", len(net.connections))
    print(net.connections)
    assert len(net.connections) == length**2, "Wrong length of conns vector"
    assert np.all(net.connections == 0) == True, "Expected conns to equal zero"
    conns = net.connections
    for i in range(len(conns)):
        conns[i] = 1
    net.connections = conns

    print(net.connections)
    assert np.all(net.connections == 1) == True, "Expected conns to equal 1"

    # ActFuncs
    print("Length: ", len(net.activation_functions))
    print(net.activation_functions)
    assert len(net.activation_functions) == length, \
           "Wrong length of conns vector"
    assert np.all(net.activation_functions == net.LINEAR) == True, \
           "Expected funcs to be LINEAR"

    actfuncs = net.activation_functions
    #actfuncs = np.zeros(len(net.activation_functions))
    for i in range(len(actfuncs)):
        actfuncs[i] = net.TANH
    print(actfuncs)
    net.activation_functions = actfuncs

    print(net.activation_functions)
    assert np.all(net.activation_functions == net.TANH) == True, \
           "Expected actfuncs to be TANH"

    # oUTPUTS
    for val in xor_in:
        assert net.output(val) != 0, "Expected some output"


    # Solve XOR
    # set all conns to zero first
    for i in range(len(conns)):
        conns[i] = 0
        weights[i] = 0
    # We care only about first two neurons and output
    actfuncs[3] = net.LOGSIG
    actfuncs[4] = net.LOGSIG
    actfuncs[5] = net.LINEAR
    net.activation_functions = actfuncs

    weights[3*length + 0] = -60
    weights[3*length + 1] = 60
    weights[3*length + 2] = -30

    weights[4*length + 0] = 60
    weights[4*length + 1] = -60
    weights[4*length + 2] = -30

    weights[(length-1)*length + 3] = 1
    weights[(length-1)*length + 4] = 1

    net.weights = weights
    net.connections = conns

    for val in xor_in:
        print("In:", val, " out:", net.output(val))
        assert net.output(val) == 0, "no conns should mean zero output!"

    conns[3*length + 0] = 1
    conns[3*length + 1] = 1
    conns[3*length + 2] = 1

    conns[4*length + 0] = 1
    conns[4*length + 1] = 1
    conns[4*length + 2] = 1

    conns[(length-1)*length + 3] = 1
    conns[(length-1)*length + 4] = 1

    net.connections = conns

    print(conns)

    for val in xor_in:
        print("In:", val, " out:", net.output(val))
        if sum(val) != 1:
            assert net.output(val) < 0.1, "xor solution doesnt work"
        else:
            assert net.output(val) > 0.9, "xor solution doesnt work"


    # Pickle test
    import pickle

    state = pickle.dumps(net, -1)

    net2 = pickle.loads(state)

    # Make sure it's the same
    assert np.all(net.weights == net2.weights), "weights don't match"

    assert np.all(net.connections == net2.connections), "conns don't match"

    assert np.all(net.activation_functions == net2.activation_functions),\
       "functions don't match"

def test_gennetwork():
    from ann import geneticnetwork

    net = geneticnetwork(2, 4, 1)

    xor_in, xor_out = getXOR()

    net.generations = 1000
    net.crossover_chance = 0.8
    net.connection_mutation_chance = 0.2
    net.activation_mutation_chance = 0.2
    net.crossover_method = net.CROSSOVER_UNIFORM
    net.fitness_function = net.FITNESS_MSE
    net.selection_method = net.SELECTION_TOURNAMENT

    net.learn(xor_in, xor_out)

    for val in xor_in:
        print("In:", val, " out:", net.output(val))
        if sum(val) != 1:
            assert net.output(val) < 0.1, "xor solution doesnt work"
        else:
            assert net.output(val) > 0.9, "xor solution doesnt work"

    print(net)
    print(dir(net))

def test_rpropnetwork():
    from ann import rpropnetwork

    net = rpropnetwork(2, 8, 1)


    # Need to connect it
    l = net.input_count + net.hidden_count + net.output_count + 1
    weights = net.weights
    conns = net.connections
    act = net.activation_functions
    # Stop before output as it is included
    for i in range(l-1):
        # connect hidden to inputs and bias
        weights[l * i: l * i + 3] = np.random.normal()
        conns[l * i: l * i + 3] = 1
        act[i] = net.TANH

    #Output
    weights[l * (l-1):] = np.random.normal(size=l)
    conns[l * (l-1):] = 1
    act[l-1] = net.LOGSIG

    net.weights = weights
    net.connections = conns
    net.activation_functions = act

    #print(net.weights)
    #print(net.connections)
    #print(net.activation_functions)

    xor_in, xor_out = getXOR()

    net.max_error = 0.001
    net.max_epochs = 1000

    net.error_function = net.ERROR_MSE

    net.learn(xor_in, xor_out)

    for val in xor_in:
        print("In:", val, " out:", net.output(val))
        if sum(val) != 1:
            assert net.output(val) < 0.1, "xor solution doesnt work"
        else:
            assert net.output(val) > 0.9, "xor solution doesnt work"

    print(net)
    print(dir(net))


if __name__ == "__main__":
    test_import()
    test_gennetwork()
    test_matrixnetwork()
    test_rpropnetwork()

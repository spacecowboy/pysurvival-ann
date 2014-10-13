
Data set
--------

Let's define the classical XOR data

.. code:: python

    import numpy as np
    
    # Using floats (doubles) are important later
    xor = np.array([[0, 0, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]], dtype=float)
    print(xor)

.. parsed-literal::

    [[ 0.  0.  0.]
     [ 0.  1.  1.]
     [ 1.  0.  1.]
     [ 1.  1.  0.]]


The input is the first two columns, and the target is the last column.
Note that the shape of the data is important.

.. code:: python

    xor_in = xor[:, :2]
    xor_target = xor[:, -1:]
    
    print(xor_in)
    print(xor_target)

.. parsed-literal::

    [[ 0.  0.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  1.]]
    [[ 0.]
     [ 1.]
     [ 1.]
     [ 0.]]


Create a suitable neural network
--------------------------------

There are different networks depending on the training method one wishes
to use. GeneticNetwork uses a genetic algorithm as is the most flexible
when it comes to error functions. RpropNetwork uses the iRprop+ method
and is the fastest but requires a differentiable error function.

Since I will use the mean square error, I'll select Rprop.

All types share the same constructor signature, taking the number of
input, hidden and output neurons respectively:

::

    net = NETWORKTYPE(input_count, hidden_count, output_count)

.. code:: python

    from ann import rpropnetwork
    
    # Create a network matching the data, with a couple of hidden neurons
    net = rpropnetwork(xor_in.shape[1], 4, xor_target.shape[1])
    
    # Total amount of neurons (including the bias neuron, of which there is only one)
    neuron_count = net.input_count + net.hidden_count + net.output_count + 1
    # All zero connections at first
    print("Default connections")
    print(net.connections.reshape((neuron_count, neuron_count)))
    # All weights are zero too
    print("\nDefault weights")
    print(net.weights.reshape((neuron_count, neuron_count)))
    
    print("\nDefault activation functions are linear ({})".format(net.LINEAR))
    print(net.activation_functions)

.. parsed-literal::

    Default connections
    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]]
    
    Default weights
    [[ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]]
    
    Default activation functions are linear (0)
    [0 0 0 0 0 0 0 0]


Connecting the network
----------------------

By default, the network has no connections between neurons. You are able
to set both the weights and connections to values of your liking but a
convenience method is supplied for creating feedforward networks.

The connections and weights are defined as NxN matrices of ints and
doubles respectively. Activation functions can also be set on the
individual neuron level if desired using the N-length vector
activation\_functions. To do interesting stuff, make sure hidden neurons
use either tanh or logsig. These are set for you by the
connect\_feedforward method.

.. code:: python

    from ann import connect_feedforward
    # Connect in a single hidden layer (default) with logsig functions on both
    # hidden and outputs (also default)
    connect_feedforward(net)
    
    print("\n\nFeedforward connections")
    print(net.connections.reshape((neuron_count, neuron_count)))
    print("\nWeights have been randomized and normalized to suitable ranges")
    print(net.weights.reshape((neuron_count, neuron_count)))
    
    print("\nActivation functions are now changed to logsig ({})".format(net.LOGSIG))
    print(net.activation_functions)
    print("\nInputs and Bias have no activation functions, or connections to other neurons")

.. parsed-literal::

    
    
    Feedforward connections
    [[0 0 1 0 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [1 1 1 0 0 0 0 0]
     [1 1 1 0 0 0 0 0]
     [1 1 1 0 0 0 0 0]
     [1 1 1 0 0 0 0 0]
     [0 0 1 1 1 1 1 0]]
    
    Weights have been randomized and normalized to suitable ranges
    [[  2.34390302e-01   1.48599413e-01  -1.27586706e+00   6.63885407e-01
       -9.57722056e-01  -5.26258883e-01  -1.17054438e+00   2.05176069e+00]
     [  2.10684640e+00  -1.20992258e+00  -7.87622229e-01   2.65956108e-01
        3.79281410e-01   1.96161791e+00   1.38639707e-02  -3.90191710e-01]
     [  6.94340032e-01   5.64623053e-01   3.62250147e-01  -2.88287230e-01
        1.26778449e+00   9.91237329e-01  -7.30969473e-01   5.56962499e-01]
     [  4.56087358e-01  -5.12488290e-01   3.14243522e-02   7.97542099e-01
        4.44247884e-01   2.58191511e-01   2.70880575e-01  -1.16556508e+00]
     [ -9.70020918e-01   1.03284643e-03  -2.89462361e-02   1.16868670e+00
        8.80225813e-01  -5.49047973e-02   1.33169280e+00   4.89071263e-02]
     [ -1.73598733e-01   2.02388658e-01  -6.24012608e-01   7.82505067e-02
       -1.22489562e-01   1.68608653e-01   1.21853670e+00   4.33320502e-02]
     [ -2.98344967e-01  -2.50058977e-01  -4.51596056e-01   6.83487914e-01
       -8.25172901e-01   6.76609952e-01   1.25793199e+00   1.63677667e+00]
     [  7.29609653e-01  -4.26469333e-01  -3.48338603e-01   3.84692714e-01
        4.74986327e-01  -4.06394526e-02  -9.96815061e-02   6.60039848e-01]]
    
    Activation functions are now changed to logsig (1)
    [0 0 0 1 1 1 1 1]
    
    Inputs and Bias have no activation functions, or connections to other neurons


Training the network
--------------------

All networks have the same training method signature:

::

    net.learn(inputdata, targetdata)

Each method naturally has a couple of different parameters you can
tweak. These are set as variables on the networks themselves. As an
example, let's see what some default Rprop values are:

.. code:: python

    print("Error function:", net.error_function, net.ERROR_MSE)
    
    print("Max training iterations:", net.maxEpochs)
    print("Max error accepted for early stopping:", net.maxError)
    print("Min change in error to consider training to be done:", net.minErrorFrac)

.. parsed-literal::

    Error function: 0 0
    Max training iterations: 1000
    Max error accepted for early stopping: 0.0001
    Min change in error to consider training to be done: 0.01


Actually train it
~~~~~~~~~~~~~~~~~

.. code:: python

    net.learn(xor_in, xor_target)
Let's see what the output and error is
--------------------------------------

To ask the network to predict something, give it a row from the input
data:

::

    net.output(xrow)

.. code:: python

    outputs = []
    for x in xor_in:
        y = net.output(x)
        
        print("{:.0f} X {:.0f} = {:.1f}".format(x[0], x[1], y[0]))
        outputs.append(y)
        
    outputs = np.array(outputs)
    
    # Note that y is an array
    y.shape == (net.output_count,)

.. parsed-literal::

    0 X 0 = 0.0
    0 X 1 = 1.0
    1 X 0 = 1.0
    1 X 1 = 0.0




.. parsed-literal::

    True



Mean square error is defined as:

.. math::  e = \frac{1}{N} \sum_i^N (\tau_i - y_i)^2 

The package however divides by two, and neglects the N-term, to make the
differential :math:`(\tau_i - y_i)` instead of:

.. math::  \frac{de}{dy_i} = \frac{2}{N}(\tau_i - y_i) 

Just to remove some calculations from the process. It has no impact on
training.

.. code:: python

    # Mean square error then
    e = np.sum((xor_target - outputs)**2) / len(xor_target)
    print("MSE: {:.6f}".format(e))
    
    # Can also ask the package to calculate it for us
    from ann import get_error
    
    e = get_error(net.ERROR_MSE, xor_target, outputs)
    
    # This is not summed for us as it is used in training piece by piece
    print("MSE: {:.6f}".format(2 * e.sum()/len(xor_target)))

.. parsed-literal::

    MSE: 0.000030
    MSE: 0.000030


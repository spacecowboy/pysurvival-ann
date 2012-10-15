#!/bin/env python

import numpy as np
import ann

x = np.genfromtxt('squares.txt', skip_header=1, usecols=range(10))

y = np.genfromtxt('squares.txt', skip_header=1, usecols=(14, 10))

# Can now access columns by name, for example
# data['X8']
# data[24]['censtime']

net = ann.getSingleLayerGenSurv(10, 20)

net.generations = 200
net.populationSize = 50

print("net.learn(x, y) to start training...")

net.learn(x, y)


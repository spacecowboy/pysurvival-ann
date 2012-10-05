import ann
import numpy as np

x = ann.ffnetwork(2,1,1)

print x.numOfInputs
print x.numOfHidden
print x.numOfOutputs

print x.output(np.array([2.0, 3.0]))

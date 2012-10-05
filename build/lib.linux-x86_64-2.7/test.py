import ann

x = ann.ffnetwork(2,4,3)

print x.numOfInputs
print x.numOfHidden
print x.numOfOutputs

print x.output([2.0, 3.0])

del x

x = ann.rpropnetwork(2, 5, 1)

print x.numOfInputs
print x.numOfHidden
print x.numOfOutputs

print x.output([1.0, 3.0])

del x

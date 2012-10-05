import ann

x = ann.ffnetwork(2,4,1)

print x.numOfInputs
print x.numOfHidden
print x.numOfOutputs

print x.output([2.0, 3.0])

del x

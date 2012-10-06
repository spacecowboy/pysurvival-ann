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

print "Before training"
print x.output([0.0, 0.0])
print x.output([0.0, 1.0])
print x.output([1.0, 0.0])
print x.output([1.0, 1.0])

data = [[0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]]

targets = [[0.0], [1.0], [1.0], [0.0]]

print "training"

x.learn(data, targets)

print "after training"

print x.output([0.0, 0.0])
print x.output([0.0, 1.0])
print x.output([1.0, 0.0])
print x.output([1.0, 1.0])


del x

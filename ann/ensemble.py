'''Ensembles of ANNs. Class definition and related functions.'''

import random
import numpy as np

def sample_wr(population, k):
    '''Selects k random elements (with replacement) from a population.
    Returns an array of indices'''
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [_int(_random() * n) for i in range(k)]

print(sample_wr(range(10), 20))

def bagging(data):
    '''Samples len elements (with replacement) from data and returns a view of those elements.'''
    return data[sample_wr(data, len(data))]

class Ensemble(object):
    '''Holds a list of any ANNs. Learn() uses bagging to give each network a unique data set.
    Bagging can be disabled by setting the attribute, then each network is trained on the
    entire dataset.'''

    def __init__(self, networks):
        '''Takes an (non-empty) iterable of networks.'''
        self.networks = list(networks)
        if len(self.networks) < 1:
            raise ValueError("List of networks must be non empty")

        self.use_bagging = True

    def __len__(self):
        return len(self.networks)

    def __getattr__(self, name):
        '''Intercept any method calls, if learn, then wrap it. Otherwise let each element evalute it.'''
        if name == "learn" and self.use_bagging:
            return lambda *args, **kwargs: self._learn_bagged(*args, **kwargs)

        if hasattr(self.networks[0], name):
            if hasattr(getattr(self.networks[0], name), '__call__'):
                #it's a function, wrap it
                return lambda *args, **kwargs: self._wrap(name, args, kwargs)
            else:
                return np.array([getattr(net, name) for net in self.networks])
        raise AttributeError(name)

    def _wrap(self, funcname, args, kwargs):
        '''Simply let each network evaluate the function and return a list of all results.'''
        result = []
        for net in self.networks:
            func = getattr(net, funcname)
            #if type(func) == MethodType:
            result.append(func(*args, **kwargs))
            #else:
            #    print(type(func))
            #    result.append(func(net, *args, **kwargs))
        return np.array(result)

    def _learn_bagged(self, trndata, targets, *args, **kwargs):
        '''Intercept the data and create slightly different data sets using bagging for each network'''
        result = []
        for net in self.networks:
            # Create new data using bagging. Combine the data into one array
            baggeddata = bagging(np.column_stack([trndata, targets]))
            result.append(net.learn(baggeddata[:, :-1], np.column_stack([baggeddata[:, -1]]), *args, **kwargs))
        return np.array(result)

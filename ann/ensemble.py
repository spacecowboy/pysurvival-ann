'''Ensembles of ANNs. Class definition and related functions.'''

from __future__ import division
import random
import numpy as np

def sample_wr(population, k):
    '''Selects k random elements (with replacement) from a population.
    Returns an array of indices.
    '''
    return np.random.randint(0, len(population), k)

def bagging(data, count=None):
    '''Samples len elements (with replacement) from data and returns a view of those elements.'''
    if count is None:
        count = len(data)
    return data[np.random.randint(0, len(data), count)]

class Ensemble(object):
    '''Holds a list of any ANNs. Learn() uses bagging to give each network a unique data set.
    Bagging can be disabled by setting the attribute, then each network is trained on the
    entire dataset.

    You can also set the bagging_limit attribute (None, or positive integer).
    None means a bagged set equal to the given data set is used.
    If a positive integer is specified, then that will be the number
    of samples drawn from the dataset. Useful if you have really big
    datasets and want to increase speed.
    '''

    def __init__(self, networks):
        '''Takes an (non-empty) iterable of networks.'''
        self.networks = list(networks)
        if len(self.networks) < 1:
            raise ValueError("List of networks must be non empty")

        self.use_bagging = True
        self.bagging_limit = None

    def __len__(self):
        return len(self.networks)


    def __getattr__(self, name):
        """Intercepts method calls and properties. The goal is to wrap such
        calls and return a list like [net1_result, net2_result, ...].

        Learn is the only method that receives special treatment because it is
        wrapped differently depending on if bagging is enabled or not.

        Other than that, all methods and properties are intercepted except:

        Properties and methods which exist for the ensemble. In that case, this
        method is never called by Python.

        Any method or property whose name starts with an underscore '_'. This is
        due to python's special handling of those methods and wrapping them is
        unwanted, it breaks pickling and who know's what else."""

        if name == "learn" and self.use_bagging:
            return lambda *args, **kwargs: self._learn_bagged(*args, **kwargs)
        # Do NOT wrap underscore methods. That would break pickling and a bunch
        # of other stuff.
        elif name.startswith('_') or not hasattr(self.networks[0], name):
            raise AttributeError(("'{}' object has no attribute " +
                                  "'{}'").format(self.__class__.__name__, name))

        # Networks not allowed to be empty, so safe
        else:# hasattr(self.networks[0], name):
            if hasattr(getattr(self.networks[0], name), '__call__'):
                # It's a function, wrap it
                return lambda *args, **kwargs: self._wrap(name, args, kwargs)
            else:
                # It's a property
                return np.array([getattr(net, name) for net in self.networks])


    def _wrap(self, funcname, args, kwargs):
        '''Simply let each network evaluate the function and return a list of
        all results.'''
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
        '''Intercept the data and create slightly different data sets using
        bagging for each network'''
        result = []
        for net in self.networks:
            # Create new data using bagging. Combine the data into one array
            baggeddata = bagging(np.column_stack([trndata, targets]), count=self.bagging_limit)
            tl = len(targets[0])
            tc = list(reversed([(-1 -x) for x in range(tl)]))
            result.append(net.learn(baggeddata[:, :-tl], np.column_stack([baggeddata[:, tc]]), *args, **kwargs))
        return np.array(result)

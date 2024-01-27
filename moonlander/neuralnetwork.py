# Daniel Cavalcanti Jeronymo
# Neural network library with layer architecure
# For teaching purposes only, this is seriously suboptimized, especially "deserialize"
#

from scipy.optimize import basinhopping, minimize, differential_evolution
import skimage.measure
from scipy import signal
from scipy import special
from sklearn import metrics
import numpy as np
import time

import mock
import pygmo as pg


########################
# Activation functions
########################
# Step function
def binary(x):
    return (x > 0.5)*1

# Self explanatory
def linear(x):
    return x

# Rectified linear unit
def relu(x):
    return (x > 0)*x

def softmax(x):
    return special.softmax(x)

########################

########################
# Layers
########################
# inputs is an Ni array
# weights is a Ni x Np matrix, Ni is number of inputs and Np is number of perceptrons
def perceptronCalculate(inputs, weights, activationFunction):
    #extendedInput = np.concatenate((inputs, [1])) # add bias input
    extendedInput = np.hstack( (inputs, [1])  ) 
    return activationFunction(extendedInput @ weights) 

# inputs is either a N1*N2 array or (N1,N2)
def conv2Calculate(inputs, layer):
    convolved = signal.convolve2d(inputs.reshape(layer["inshape"]), layer["weights"], mode='valid')
    subsampled = skimage.measure.block_reduce(convolved, layer["subshape"], np.max)

    return subsampled.flatten()

def layerCalculate(inputs, layer):
    weights = layer["weights"]

    if layer["type"] == "dense":
        return perceptronCalculate(inputs, weights, layer["function"])
    elif layer["type"] == "conv2":
        return conv2Calculate(inputs, layer)
    else:
        print("LAYER CALCULATE ERROR! Unknown type ", layer["type"])
        exit(1)
        return None

# nin defines number of inputs for each neuron in this layer
# nout defines the number of neurons
def layerDenseCreate(nin, nout, activationFunction):
    l = {
    "type": "dense",
    "nin" : nin,
    "nout" : nout,
    "shape" : (nin+1,nout),
    "nweights" : (nin + 1)*nout,
    "weights" : np.random.random_sample((nin+1, nout)),#np.zeros((nin+1, nout)),
    "function" : activationFunction
    }

    return l

def layerConv2Create(inShape, subShape, kernelShape):
    l = {
    "type" : "conv2",
    "nin" : inShape[0]*inShape[1],
    "nout" : inShape[0]*inShape[1] - (kernelShape[0]*kernelShape[1] - 1),
    "shape" : kernelShape,
    "inshape" : inShape,
    "subshape" : subShape,
    "nweights" : kernelShape[0]*kernelShape[1],
    "weights" : np.random.random_sample(kernelShape)#np.zeros(kernelShape)
    }

    return l

########################
# Network
########################
def networkCalculate(inputs, layers):
    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, 0)

    outputs = []
    for x in inputs:
        currentInputs = x
        for layer in layers:
            #if len(currentInputs) != layer["nin"]:
            if currentInputs.size != layer["nin"]:
                print("NETWORK ERROR: Mismatched input shape ", currentInputs.shape, " and layer shape ", layer["shape"])
                print("NETWORK ERROR: Layer data: ", layer)
                exit(1)
                
            output = layerCalculate(currentInputs, layer)
            currentInputs = output

        outputs.append(output)

    noutputs = len(outputs)
    ndims = len(outputs[0])
    return np.array(outputs).reshape(noutputs, ndims)

# Extracts all weights from layers in this network
def networkSerialize(layers):
    weights = []
    for layer in layers:
        weights.append(layer["weights"].flatten())
    
    # concatenate is used to join all arrays in 'weights'
    return np.concatenate(weights).flatten()

# Inserts weights into layers
def networkDeserialize(weights, layers):
    offset = 0
    for layer in layers:
        w = weights[offset:offset+layer["nweights"]]
        layer["weights"] = np.reshape(w, layer["shape"])
        
        offset += layer["nweights"]

    return layers

########################
# Network training
########################
# cross entropy loss function
def CrossEntropy(y, yp):
    return metrics.log_loss(y, yp)

# mean squared error loss function
def mse(y, yp):
    return np.mean((y - yp)**2)

def loss(x, y, layers):
    return mse(y.flatten(), networkCalculate(x, layers).flatten())
    #return CrossEntropy(y.flatten(), networkCalculate(x, layers).flatten())

# helper function for minimizers
def loss2(w, x, y, layers):
    layers = networkDeserialize(w, layers)
    return loss(x, y, layers)

# Loss function structured as LossProblem for pygmo minimization
class LossProblem:
    def __init__(self, dim, x, y, layers):
        self.dim = dim
        self.x = x
        self.y = y
        self.layers = layers

        self.bounds = ([-1] * self.dim, [1] * self.dim)

    def fitness(self, x):
        return [loss2(x, self.x, self.y, self.layers)]

    def get_bounds(self):
        return self.bounds
        

# stochastic search for values of weights and bias until MSE is zero
# WARNING: this WILL lock if search space is too big or no solution can be found
def trainRandom(x, y, layers, itmax=10000):
    # define network total size
    weightCount = networkSerialize(layers).size
    
    bestx = np.zeros(weightCount)
    bestf = 1e20
    
    t0 = time.perf_counter()

    for _ in range(itmax):
        w = np.random.normal(0, 2, weightCount)
        layers = networkDeserialize(w, layers)

        res = loss(x, y, layers)

        if res < bestf:
            bestf = res
            bestx = w

        if res == 0:
            break

    t1 = time.perf_counter()
    print('TRAINING COMPLETE! Solution time: {}'.format(t1-t0))
    print("Best X: ", bestx)
    print("Best: ", bestf)
        
    return bestx

def trainMinimize(x, y, layers, method='min'):
    # define network total size
    weightCount = networkSerialize(layers).size

    # random initial weights
    x0 = np.random.normal(0, 2, weightCount)
    xlimits = [(-1,1)]*weightCount
    args = (x, y, layers)
    minimizer_kwargs = {"method": "SLSQP", "bounds": xlimits, "args": args} # basinhopping

    t0 = time.perf_counter()

    if method == 'min':
        #result = minimize(loss2, x0, method='BFGS', bounds=xlimits, tol=1e-10, args=args)
        result = minimize(loss2, x0, args=args)
    elif method == 'DE':
        result = differential_evolution(loss2, xlimits, updating='deferred', workers=-1, args=args)
    elif method == 'basin':
        result = basinhopping(loss2, x0, minimizer_kwargs=minimizer_kwargs, T=0.1, niter=200, interval=10)

    t1 = time.perf_counter()
    print('TRAINING COMPLETE! Solution time: {}'.format(t1-t0))
    #print(result)
    print("Best X: ", result.x)
    print("Best: ", result.fun)

    return result.x

def trainMetaheuristic(x, y, layers, method='cmaes'):
    weightCount = networkSerialize(layers).size

    t0 = time.perf_counter()

    algo = pg.algorithm(pg.cmaes(gen = 100, sigma0=0.3))
    #algo = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))

    lossProb = LossProblem(weightCount, x, y, layers)
    prob = pg.problem(lossProb)

    pop = pg.population(prob, 100)
    pop = algo.evolve(pop)

    #print(prob)

    result = mock.Mock()
    result.x = pop.get_x()[pop.best_idx()]
    result.fun = pop.get_f()[pop.best_idx()]

    t1 = time.perf_counter()
    #felipe:: removidos os prints
    #print('TRAINING COMPLETE! Solution time: {}'.format(t1-t0))

    #print("Best X: ", result.x)
    #print("Best: ", result.fun)

    return result.x

def train(x, y, layers, method='min'):
    if method == 'random':
        return trainRandom(x, y, layers)
    elif method == 'min':
        return trainMinimize(x, y, layers)
    elif method == 'DE':
        return trainMinimize(x, y, layers, method='DE')
    elif method == 'basin':
        return trainMinimize(x, y, layers, method='basin')
    elif method == 'cmaes':
        return trainMetaheuristic(x, y, layers, method='cmaes')
    else:
        print("TRAIN ERROR: Unknown method ", method)
        exit(1)
        return None

def main():
    # Training sets
    # binary inputs
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # AND outputs
    yAND = np.array([0, 0, 0, 1])
    # NAND outputs
    yNAND = np.array([1, 1, 1, 0])
    # OR outputs
    yOR = np.array([0, 1, 1, 1])
    # XOR outputs
    yXOR = np.array([0, 1, 1, 0])

    # Network with two layers
    layerNANDOR = layerDenseCreate(2, 2, relu) # two inputs, two neurons with relu function
    layerAND = layerDenseCreate(2, 1, linear) # two inputs from previous layer, 1 linear output
    layers = [layerNANDOR, layerAND]

    w = train(x, yXOR, layers, method='cmaes')

    layers = networkDeserialize(w, layers)

    print(networkCalculate(x, layers))

    ########




if __name__ == "__main__":
    main()
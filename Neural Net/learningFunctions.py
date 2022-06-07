import numpy as np

# ACTIVATION FUNCTIONS    
## Gaussian
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dSigmoid(x):
    a = (1+np.exp(-x))**2
    return ((np.exp(-x))/a)

## Rectified Linear Unit
def reLu(x, threshold = 0):
    k = x
    if x < threshold:
        k = 0
    return k

def dReLu(x, threshold = 0):
    k = 1
    if x < threshold:
        k = 0
    return k

## Leaky Rectified Linear Unit
def leakyReLu(x, threshold = 0, reduction = 0.1):
    k = x
    if x < threshold:
        k = reduction * k
    return k

def dLeakyReLu(x, threshold = 0, reduction = 0.1):
    k = 1
    if x < threshold:
        k = reduction * k
    return k

## tanh
def tanh(x):
    numer = 1 - np.exp(-2*x)
    denom = 1 + np.exp(-2*x)
    return numer/denom

def dTanh(x):
    intermediate = (1 - (1/(1+np.exp(-2*x))))
    return tanh(x) * intermediate

## Swish
def swish(x):
    return 2 * x * sigmoid(x)

def dSwish(x):
    first = 2 * x * dSigmoid(x)
    second = 2 * sigmoid(x)
    return first + second

## Softmax
def softmax(x):
    return np.exp(x)/np.exp(x).sum(axis = 0)
    
# Error Functions
## Sum of Squared Errors
def sumSqError(y, y_hat):
    diff = y_hat - y
    return np.sum(np.dot(diff, diff))

## Negative Log Likelihood
def logLike(y, y_hat):
    return -(np.dot(y, np.log(y_hat)) + np.dot((1 - y),np.log(1 - y_hat))).sum()

## Cross Entropy
def crossEntropy(y, y_hat):
    return -np.multiply(y, np.log(y_hat)).sum()

## General Differential for Error terms
def dError(y, y_hat):
    return y_hat - y

def differentiate(x, activation):
    if activation == "Sigmoid":
        return np.vectorize(dSigmoid)(x)
    elif activation == "ReLU":
        return np.vectorize(dReLu)(x)
    elif activation == "Leaky ReLU":
        return np.vectorize(dLeakyReLu)(x)
    elif activation == "Tanh":
        return np.vectorize(dTanh)(x)
    else:
        return np.vectorize(dSwish)(x)

def activate(x, activation):
    if activation == "Sigmoid":
        return np.vectorize(sigmoid)(x)
    elif activation == "ReLU":
        return np.vectorize(reLu)(x)
    elif activation == "Leaky ReLU":
        return np.vectorize(leakyReLu)(x)
    elif activation == "Tanh":
        return np.vectorize(tanh)(x)
    else:
        return np.vectorize(swish)(x)
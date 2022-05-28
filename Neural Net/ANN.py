import numpy as np
import math

class ANN:
    def __init__(self, inLayer, hiddenLayers, outLayer, learning_rate):
        self.inputSize = inLayer
        self.hiddenSize = hiddenLayers
        self.outputSize = outLayer
        self.lr = learning_rate
        
        self.gradients = []
        self.gradients.append(np.ones((inLayer, hiddenLayers[0])))
        if len(hiddenLayers) > 1:
            for layer in range(len(hiddenLayers) - 1):
                self.gradients.append(np.ones((hiddenLayers[layer], hiddenLayers[layer + 1])))
        self.gradients.append(np.ones((hiddenLayers[-1], outLayer)))

        self.values = []
        for layers in range(len(hiddenLayers)+2):
            self.values.append(np.array([]))


    def trainModel(self, x, y, epochs):
        for cycle in range(epochs):
            for k in range(len(x)):
                self.forwardProp(x[k])
                self.backProp(y[k])

    def forwardProp(self, data):
        self.values[0] = np.array(data)
        for grad in range(len(self.gradients)):
            valIndex = grad + 1
            prev = self.values[grad]
            w = self.gradients[grad]
            temp = np.dot(prev, w)
            result = self.sigmoid(temp)
            self.values[valIndex] = result


    def backProp(self, y):
        deltas = []
        corrections = []
        dError = self.values[-1]-y
        for i in range(len(self.gradients) - 1, -1, -1) :
            if len(deltas) == 0:
                dError.reshape((1, self.outputSize))
                dSig = self.dSigmoid(np.dot(self.values[i], self.gradients[i]))
                dSig.reshape((1, dSig.shape[0]))
                delta = np.multiply(dError.T, dSig)
                delta.reshape((1,delta.shape[0]))
                deltas.append(delta)
                z = self.values[i]
                if len(z.shape) == 1:
                    z = self.values[i].reshape((1,self.values[i].shape[0]))
                correction = np.dot(z.T, delta)
                corrections.append(correction)

            else:
                delta = deltas[-1]
                if len(delta.shape) == 1:
                    delta = delta.reshape((1,delta.shape[0]))
                dSig = self.dSigmoid(np.dot(self.values[i], self.gradients[i]))
                if len(dSig.shape) == 1:
                    dSig = dSig.reshape((1,dSig.shape[0]))
                delta = np.dot(delta, self.gradients[i+1].T)
                delta = delta * dSig
                deltas.append(delta)
                lastVal = self.values[i]
                if len(lastVal.shape) == 1:
                    lastVal = self.values[i].reshape((1, len(self.values[i])))
                correction = np.dot(delta.T,lastVal)
                corrections.append(correction.T)
        corrections.reverse()
        for i in range(len(corrections)):
            self.gradients[i] = self.gradients[i] - self.lr * corrections[i]
    
    # ACTIVATION FUNCTIONS    
    ## Gaussian
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dSigmoid(self, x):
        a = (1+np.exp(-x))**2
        return ((np.exp(-x))/a)
    
    ## Rectified Linear Unit
    def reLu(x, threshold):
        k = x
        if x < threshold:
            k = 0
        return k
    
    def dReLu(x, threshold):
        k = 1
        if x < threshold:
            k = 0
        return k
    
    ## Leaky Rectified Linear Unit
    def leakyReLu(x, threshold, reduction):
        k = x
        if x < threshold:
            k = reduction * k
        return k
    
    def dLeakyReLu(x, threshold, reduction):
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
        return
    
    ## Swish
    def swish(x):
        return 2 * x * self.sigmoid(x)
    
    def dSwish(x):
        return
    
    # Softmax
    def softmax(x):
        return np.exp(x)/np.exp(x).sum(axis = 0)
        

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
    
    def sigmoid(self, x):
        return 1/(1+math.e**(-x))

    def dSigmoid(self, x):
        a = (1+math.e**(-x))**2
        return ((math.e**(-x))/a)

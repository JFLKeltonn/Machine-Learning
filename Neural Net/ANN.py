import numpy as np
import learningFunctions as lf

class ANN:
    def __init__(self, inLayer, hiddenLayers, outLayer, learning_rate, actFunc = "Sigmoid"):
        self.inputSize = inLayer
        self.hiddenSize = hiddenLayers
        self.outputSize = outLayer
        self.lr = learning_rate
        self.activation = actFunc
        
        
        self.gradients = []
        self.gradients.append(np.random.rand(inLayer, hiddenLayers[0]))
        if len(hiddenLayers) > 1:
            for layer in range(len(hiddenLayers) - 1):
                self.gradients.append(np.random.rand(hiddenLayers[layer], hiddenLayers[layer + 1]))
        self.gradients.append(np.random.rand(hiddenLayers[-1], outLayer))

        self.values = []
        self.preAct = []
        for layers in range(len(hiddenLayers)+2):
            self.values.append(np.array([]))
            self.preAct.append(np.array([]))


    def trainModel(self, x, y, epochs):
        print("Starting Model Training")
        tenPercent = epochs // 10
        progress = 0
        for cycle in range(epochs):
            for k in range(len(x)):
                self.forwardProp(x[k])
                self.backProp(y[k])
            if (1+cycle) % tenPercent == 0:
                progress += 1
                print("Training is {prog}% done.".format(prog = progress * 10))

    def forwardProp(self, data):
        self.values[0] = np.array(data)
        self.preAct[0] = np.array(data)
        for grad in range(len(self.gradients)):
            valIndex = grad + 1
            nodes = self.values[grad]
            weights = self.gradients[grad]
            self.preAct[valIndex] = np.dot(nodes, weights)
            result = lf.activate((self.preAct[valIndex]),self.activation)
            self.values[valIndex] = result


    def backProp(self, y):
        deltas = []
        corrections = []
        dError = lf.dError(y, self.preAct[-1])
        for i in range(len(self.gradients) - 1, -1, -1) :
            if len(deltas) == 0:
                dError.reshape((1, self.outputSize))
                # actDiff = lf.differentiate(self.preAct[i + 1], self.activation)
                # delta = np.multiply(dError.T, actDiff)
                # delta.reshape((1,delta.shape[0]))
                delta = dError
                deltas.append(delta)
                z = self.values[i]
                if len(z.shape) == 1:
                    z = self.values[i].reshape((1,self.values[i].shape[0]))
                correction = np.dot(z.T, delta)
                corrections.append(correction)

            else:
                delta = deltas[-1]
                actDiff = lf.differentiate(self.preAct[i + 1], self.activation)
                delta = np.dot(delta,self.gradients[i+1].T)
                delta = delta * actDiff
                deltas.append(delta)
                lastVal = self.values[i]
                correction = np.dot(delta.T,lastVal)
                corrections.append(correction.T)
        corrections.reverse()
        for i in range(len(corrections)):
            self.gradients[i] = self.gradients[i] - self.lr * corrections[i]
    
    def predict(self, x):
        self.forwardProp(x)
        return self.preAct[-1]


def test():
    x = [1,2,3,4,5,6]
    y = [2*k for k in x]
    hiddenLayers = [random.randint(1,10) for i in range(4)]
    alpha = 0.001
    epoch = 10000000
    print("==========================================")
    print("Starting test. Test Parameters are as follows:")
    print("INPUT: {x1} \nTARGET: {y1}\nLAYERS: {lay}".format(x1 = x, y1 = y, lay = hiddenLayers))
    print("LEARNING RATE: {lr} \nEPOCHS: {ep}".format(lr = alpha, ep = epoch))
    print("==========================================")
    a = ANN(1,hiddenLayers,1,alpha)
    a.trainModel(x, y, epoch)
    print("==========================================")
    results = []
    for i in x:
        results.append(a.predict(i)[0][0])
    print("PREDICTED RESULTS: {r}".format(r = results))
    print("EXPECTED TARGET: {y1}".format(y1 = y))
    print("ERROR: {err}".format(err = lf.sumSqError(np.array(y), np.array(results))))

test()
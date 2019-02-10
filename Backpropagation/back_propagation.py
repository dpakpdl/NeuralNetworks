import numpy as np
import math
from random import shuffle, choice


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, Inum, Hnum, Onum):
        self.Inum = Inum
        self.Hnum = Hnum
        self.Onum = Onum

        self.lr = 0.1

        self.weightsIH = np.random.rand(self.Hnum, self.Inum) * 2 - 1
        self.weightsHO = np.random.rand(self.Onum, self.Hnum) * 2 - 1

        self.biasH = np.random.rand(self.Hnum) * 2 - 1
        self.biasO = np.random.rand(self.Onum) * 2 - 1

        W1 = [[0.5, 0.4],
              [.9, 1.0]]

        self.weightsIH = np.array(W1)
        self.biasH = np.array([0.8, -0.1])
        W2 = [[-1.2, 1.1]]
        self.weightsHO = np.array(W2)
        self.biasO = np.array([0.3])

    def feedForward(self, inputs):
        hidden = np.dot(self.weightsIH, np.array(inputs))
        hidden = hidden + self.biasH
        hidden = sigmoid(hidden)
        outputs = np.dot(self.weightsHO, hidden)
        outputs = outputs + self.biasO
        outputs = sigmoid(outputs)

        return outputs

    def train(self, inputs, targets):
        """
        NOTE : Always deal with column vectors as you do in maths.
        """
        # Feed Forward
        hidden = np.dot(self.weightsIH, np.array(inputs))
        hidden = hidden - self.biasH
        hidden = sigmoid(hidden)
        print "hidden output %s %s" %(hidden, inputs)
        outputs = np.dot(self.weightsHO, hidden)
        outputs = outputs - self.biasO
        outputs = sigmoid(outputs)
        print "output %s target %s" % (outputs, targets)
        # Calculate errors
        errorsO = np.array(targets) - outputs
        print "error %s" %errorsO
        errorsO = errorsO[:, np.newaxis]  # errorsO is a column now

        # Calculate gradients with derivitive of sigmoid
        gradientsO_ = dsigmoid(outputs)
        # Convert gradientsO also to column vector before taking product
        gradientsO_ = gradientsO_[:, np.newaxis] * errorsO  # Hadamard product to get a new column vector
        print "delta_o %s" % gradientsO_
        gradientsO = gradientsO_ * self.lr
        # Calculate deltas
        hiddenT = hidden[np.newaxis]  # hidden is a column now
        weightsHODeltas = np.dot(gradientsO, hiddenT)
        print "wt change %s" % weightsHODeltas
        # Adjust weights by deltas
        # self.weightsHO = self.weightsHO + weightsHODeltas

        # Adjust bias by gradients
        bias = gradientsO.reshape(self.biasO.shape)
        print "bias change %s" % bias
        self.biasO = self.biasO + gradientsO.reshape(self.biasO.shape)
        print self.biasO
        # Hidden layer
        errorsH = np.dot(np.transpose(self.weightsHO),
                         gradientsO_)  # You had a conceptual mistake here. You don't incoporate learning rate here

        print errorsH
        # Calculate gradients with derivitive of sigmoid
        gradientsH = dsigmoid(hidden)
        print gradientsH
        gradientsH = gradientsH[:, np.newaxis] * errorsH
        print "deltas h %s" % gradientsH
        gradientsH = gradientsH * self.lr

        # Calculate deltas
        inputsT = np.array(inputs)[np.newaxis]
        weightsIHDeltas = np.dot(gradientsH, inputsT)
        print weightsIHDeltas
        # Adjust weights by deltas
        self.weightsIH = self.weightsIH + weightsIHDeltas
        print self.weightsIH
        # Adjust bias by gradients
        print "bias change %s" % gradientsH.reshape(self.biasH.shape)
        self.biasH = self.biasH + gradientsH.reshape(self.biasH.shape)
        self.weightsHO = self.weightsHO + weightsHODeltas
        print self.weightsHO
        print self.biasH



def main():
    nn = NeuralNetwork(2, 2, 1)
    print nn.weightsIH
    print nn.biasH
    print nn.weightsHO
    print nn.biasO
    dataset = [
        {
            "inputs": [0, 0],
            "outputs": 0
        },
        {
            "inputs": [0, 1],
            "outputs": 1
        },
        {
            "inputs": [1, 0],
            "outputs": 1
        },
        {
            "inputs": [1, 1],
            "outputs": 0
        }
    ]

    for x in range(2):
        nn.train(dataset[3]["inputs"], dataset[3]["outputs"])
        # for data in dataset:
        #     nn.train(data["inputs"], data["outputs"])
        # shuffle(dataset)


if __name__ == "__main__":
    main()
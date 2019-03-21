import numpy as np


def perceptron_test(features, weights, z):
    y_pred = []
    for i in xrange(0, len(features - 1)):
        f = np.dot(features[i], weights)

        # activation function
        if f > z:
            yhat = 1
        else:
            yhat = 0
        y_pred.append(yhat)
    return y_pred


def get_x2(x1, weights):
    w0 = 0
    w1 = weights[0]
    w2 = weights[1]
    x2 = []
    for i in xrange(0, len(x1 - 1)):
        x2_temp = (-w0 - w1 * x1[i]) / w2
        x2.append(x2_temp)
    return x2

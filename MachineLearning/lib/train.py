import numpy as np


def perceptron_train(features, class_labels, z, eta, number_of_iterations):
    # Initializing parameters for the perceptron
    weight_vector = np.zeros(len(features[0]))  # weights
    counter = 0

    # Initializing additional parameters to compute SSE
    predicted_class_vector = np.ones(len(class_labels))  # vector for predictions
    errors = np.ones(len(class_labels))  # vector for errors (actual - predictions)
    cost_function_vector = []  # vector for the SSE cost function

    while counter < number_of_iterations:
        for i in xrange(0, len(features)):

            # summation step
            f = np.dot(features[i], weight_vector)

            # activation function
            if f > z:
                yhat = 1.
            else:
                yhat = 0.
            predicted_class_vector[i] = yhat

            # updating the weights
            for j in xrange(0, len(weight_vector)):
                weight_vector[j] = weight_vector[j] + eta * (class_labels[i] - yhat) * features[i][j]

            counter += 1

            # computing the sum-of-squared errors
        for i in xrange(0, len(class_labels)):
            errors[i] = (class_labels[i] - predicted_class_vector[i]) ** 2
        cost_function_vector.append(0.5 * np.sum(errors))

    # function returns the weight vector, and sum-of-squared errors
    return weight_vector, cost_function_vector

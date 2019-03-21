import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lib.test import perceptron_test, get_x2
from lib.train import perceptron_train


def read_dataset():
    dataset = pd.read_csv('data/medicine.csv', names=["sideEffects", "recovery", "class"])
    print dataset.head()
    feature_vectors = dataset.iloc[:, :-1].values
    class_labels = dataset.iloc[:, 2].values
    return feature_vectors, class_labels


def scatter_plot_data(feature_vectors, class_labels):
    feature_columns = ["sideEffects", "recovery"]
    classes = class_labels
    palette = {1: "red", 2: "blue"}
    class_label = {1: "Class A", 2: "Class B"}

    for (i, cla) in enumerate(set(classes)):
        xc = [p for (j, p) in enumerate(feature_vectors[:, 0]) if classes[j] == cla]
        yc = [p for (j, p) in enumerate(feature_vectors[:, 1]) if classes[j] == cla]
        cols = [palette[c] for (j, c) in enumerate(class_labels) if classes[j] == cla]
        plt.scatter(xc, yc, c=cols, label=class_label[cla], edgecolors='k')
    plt.legend(loc=1)
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.title("Effect of classes of medicines in influenza patients", fontsize=10)


def plot_sqaure_error_vs_epoch(errors):
    epoch = np.linspace(1, len(errors), len(errors))

    plt.plot(epoch, errors)
    plt.xlabel('Epoch')
    plt.ylabel('Sum-of-Squared Error')
    plt.title('Perceptron Convergence')
    plt.show()


def main():
    feature_vectors, class_labels = read_dataset()
    scatter_plot_data(feature_vectors, class_labels)
    z = 0.0  # threshold
    eta = 0.005  # learning rate
    iterations = 15000  # number of iterations

    le = LabelEncoder()
    class_labels = le.fit_transform(class_labels)
    train_features, test_features, train_labels, test_labels = train_test_split(feature_vectors, class_labels,
                                                                                test_size=0.2, random_state=0)

    weight_vector, errors = perceptron_train(train_features, train_labels, z, eta, iterations)
    predicted_labels = perceptron_test(test_features, weight_vector, z)

    print accuracy_score(test_labels, predicted_labels)
    print "The sum-of-squared erros are:"
    print errors

    # plot the decision boundary
    # 0 = w0x0 + w1x1 + w2x2
    # x2 = (-w0x0-w1x1)/w2

    min = np.min(test_features[:, 1])
    max = np.max(test_features[:, 1])
    x1 = np.linspace(min, max, 100)

    x_2 = np.asarray(get_x2(x1, weight_vector))
    plt.plot(x1, x_2)
    plt.show()

    plot_sqaure_error_vs_epoch(errors)


if __name__ == "__main__":
    main()

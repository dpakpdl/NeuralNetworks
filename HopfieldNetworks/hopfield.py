import numpy as np


def multiply_matx_transpose(matrix):
    return np.dot(matrix.T, matrix)


def hopfield(inputs, n):
    s = (n, n)
    sum_ = np.zeros(s)
    for ip in inputs:
        sum_ += multiply_matx_transpose(ip)
    return sum_ - len(inputs) * get_identity_matrix(n)


def get_identity_matrix(n):
    return np.matrix(np.identity(n), copy=False)


if __name__ == "__main__":
    inputs = np.array([[[1, 1, 1]], [[-1, -1, -1]], [[1, -1, -1]]])
    print inputs
    results = hopfield(inputs, inputs[0].shape[1])
    print results
    s = (len(inputs), inputs[0].shape[1])
    output_check = list()
    for ip in inputs:
        check = np.dot(results, ip.T)
        check[check < 0] = -1
        check[check > 0] = 1
        output_check.append(check.T)
    print np.vstack(output_check)

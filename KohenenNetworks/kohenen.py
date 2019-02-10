import numpy as np


def get_euclidian_distance(ip, weights):
    return np.linalg.norm(ip - weights)


if __name__ == "__main__":
    ip = np.array([0.52, 0.12])
    learning_rate = 0.1
    weights = np.array([[0.27, 0.81], [0.42, 0.70], [0.43, 0.21]])
    print "Initial Weights %s" %weights
    print "Inputs %s" %ip
    for i in range(3):
        print "EPOCH%s------------------------------" %(i+1)
        dist = list()
        for weight in weights:
            dist.append(round(get_euclidian_distance(ip, weight), 4))
        print "Distances %s" % dist
        val, idx = min((val, idx) for (idx, val) in enumerate(dist))
        print "Minimum distance %s at %s" %(val, idx+1)
        del_wt = np.round(learning_rate * (ip - weights[idx]), 4)
        print "del weight %s of %s" %(del_wt, idx+1)
        weights[idx] = weights[idx] + del_wt
        print weights[idx]
        print "updated weights %s" %weights

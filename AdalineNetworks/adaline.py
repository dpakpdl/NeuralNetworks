import csv
# import random


class DimensionMissMatchException(Exception):
    def __init__(self, error_message):
        super(DimensionMissMatchException, self).__init__(error_message)


def calculate_input(bias, _inputs, _weights):
    """
    :param bias: single value
    :param _inputs: vector of inputs in an epoch
    :param _weights: vector of weights in an epoch
    :return: new target
    """
    if len(_inputs) != len(_weights):
        raise DimensionMissMatchException("Weight not assigned for all inputs")
    return float(bias) + sum(map(lambda xi, wi: float(xi) * float(wi), _inputs, _weights))


def calculate_bias(old_bias, learning_rate, output, _input):
    return float(old_bias) + float(learning_rate) * (float(output) - float(_input))


def calculate_weights(weights, inputs, learning_rate, output, cal_input):
    return list(map(lambda xi, wi: float(wi) + (float(learning_rate) * (float(output) - float(cal_input)) * float(xi)), inputs, weights))


def main():
    file_path = 'truth_table.csv'
    fs = csv.reader(open(file_path, newline='\n'))
    all_row = [r for r in fs]
    rows = all_row[1:]
    # weights = [round(random.uniform(0.1, 0.4), 1) for i in range(0, 2, 1)]
    weights = [0.2, 0.3]
    learning_rate = 0.1
    bias = 0.1
    print('wt1  wt2  bias  y_in')
    for i in range(0, 1000, 1):
        print ("-"*100)
        old_bias = bias
        for row in rows:
            cal_input = round(calculate_input(bias, row[:2], weights), 2)
            weights = calculate_weights(weights, row[:2], learning_rate, row[3], cal_input)
            weights = [round(weight, 2) for weight in weights]
            bias = round(calculate_bias(bias, learning_rate, row[3], cal_input), 2)
            print (weights[0], weights[1], bias, cal_input)
        new_bias = bias
        if abs(new_bias - old_bias) < 0.001:
            break


if __name__ == "__main__":
    main()

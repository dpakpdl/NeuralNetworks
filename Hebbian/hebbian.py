import csv


def get_new_weight(_weights, _inputs, output):
    return list(map(lambda xi, wi: float(wi) + float(xi) * float(output), _inputs, _weights))


def get_new_bias(old_bias, output):
    return float(old_bias) + float(output)


def main():
    weights = [0, 0]
    bias = 0
    file_path = 'truth_table.csv'
    fs = csv.reader(open(file_path, newline='\n'))
    all_row = [r for r in fs]
    rows = all_row[1:]
    print ('x1 x2 w1 w2 b')
    for row in rows:
        weights = get_new_weight(weights, row[:2], row[3])
        bias = get_new_bias(bias, row[3])
        print (row[0], row[1], weights[0], weights[1], bias)


if __name__ == '__main__':
    main()

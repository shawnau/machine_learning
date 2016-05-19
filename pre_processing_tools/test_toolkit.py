import numpy as np


def load_data(filename):
    data_matrix = []
    f = open(filename)
    for line in f.readlines():
        line_array = line.strip().split('\t')
        data_matrix.append(line_array)
    return np.array(data_matrix).astype(np.float)


def separate_x_y(data_matrix):
    m, n = data_matrix.shape
    x = data_matrix[:, 0:(n - 1)]  # slice from 0 to n - 2
    y = data_matrix[:, n-1]  # convert to float, then to int
    y = y.astype(np.int).reshape(m, 1)
    return x, y


def split_data(filename):
    data_matrix = load_data(filename)
    m, n = data_matrix.shape

    train_size = int(0.7 * m)
    cv_size = int(0.2 * m)

    train_matrix, cv_matrix, test_matrix = \
        np.vsplit(data_matrix[np.random.permutation(m)], (train_size, train_size + cv_size))
    return train_matrix, cv_matrix, test_matrix


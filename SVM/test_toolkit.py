import numpy as np
import scipy.io as sio


def load_data(filename):
    """
    :param: filename
    :return: np.array().astype(np.float)
    reads from .txt files.
    elements are separated bt '\t'
    return an array with each corresponding line
    elements' type converted into float
    """
    data_matrix = []
    f = open(filename)
    for line in f.readlines():
        line_array = line.strip().split('\t')
        data_matrix.append(line_array)
    return np.array(data_matrix).astype(np.float)


def load_mat(filename):
    """
    :param: filename: matlab files (.mat)
    :return: dict info and data_matrix.astype(np.float)
    """
    file_content = sio.loadmat(filename)
    print('The keys are: \n')
    print(file_content.keys())
    # this part is made only for the test data 'test_data/ex6data2.mat'
    x = np.array(file_content['X']).astype(np.float)
    y = np.array(file_content['y']).astype(np.float)
    for i in range(int(y.shape[0])):
        if y[i] == 0.0:
            y[i] = -1.0
    data_matrix = np.concatenate((x, y), axis=1)
    return data_matrix


def split_data(data_matrix, ratio):
    """
    :param: data_matrix (dtape = np.array).astype(np.float), ratio(tuple)
    :return: 3 np.array()
    randomly split it into 3 parts using np.vsplit
    each part has ratio[0] : ratio[1] : 1-ratio[0]-ratio[1] columns
    """
    m, n = data_matrix.shape

    train_size = int(ratio[0] * m)
    cv_size = int(ratio[1] * m)

    train_matrix, cv_matrix, test_matrix = \
        np.vsplit(data_matrix[np.random.permutation(m)], (train_size, train_size + cv_size))
    return train_matrix, cv_matrix, test_matrix


def separate_x_y(data_matrix):
    """
    :param: np.array().astype(np.float)
    :return: x(dtape = float), y(dtape = int)
    reads an array
    returns the last column as y, others as x
    """
    m, n = data_matrix.shape
    x = data_matrix[:, 0:(n - 1)]  # slice from 0 to n - 2
    y = data_matrix[:, n-1]  # convert to float, then to int
    y = y.astype(np.int).reshape(m, 1)
    return x, y



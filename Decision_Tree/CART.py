import numpy as np


class TreeNode():
    def __init__(self, feat, val, left, right):
        feature = feat
        value = val
        left_branch = left
        right_branch = right


def split(data_matrix, feature_index, value):
    value_list = data_matrix[:, feature_index]

    # left_indexes = (array([ 8,  9, 10, 13], dtype=int64), ), which is a tuple
    # while left_indexes[0] = array([ 8,  9, 10, 13], dtype=int64), which is an array
    left_indexes = np.nonzero(value_list > value)[0]
    left_matrix = data_matrix[left_indexes]

    right_indexes = np.nonzero(value_list <= value)[0]
    right_matrix = data_matrix[right_indexes]
    return left_matrix, right_matrix

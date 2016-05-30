import numpy as np


class TreeNode():
    def __init__(self, feat, val, left, right):
        feature = feat
        value = val
        left_branch = left
        right_branch = right


# Notice: the main difference between this and ID3 splitting function
# is that CART could choose the same feature in different nodes.
def split(data_matrix, feature_index, value):
    value_list = data_matrix[:, feature_index]

    # left_indexes = (array([ 8,  9, 10, 13], dtype=int64), ), which is a tuple
    # while left_indexes[0] = array([ 8,  9, 10, 13], dtype=int64), which is an array
    left_indexes = np.nonzero(value_list <= value)[0]
    left_matrix = data_matrix[left_indexes]

    right_indexes = np.nonzero(value_list > value)[0]
    right_matrix = data_matrix[right_indexes]
    return left_matrix, right_matrix


def var_error_sum(data_list):
    return data_list.shape[0] * np.var(data_list)


def choose_feature(data_matrix, option):
    toler_error = option[0]
    toler_size = option[1]
    m, n = data_matrix.shape
    y = data_matrix[:, -1]

    if list(y).count(y[0]) == len(y):
        return None, np.mean(y)

    least_square_error = np.inf
    chosen_feature_index = 0
    chosen_feature_value = 0.0
    find_split_flag = 0
    for feature_index in range(n-1):
        for value in set(data_matrix[:, feature_index]):
            left_matrix, right_matrix = split(data_matrix, feature_index, value)
            if (left_matrix.shape[0] < toler_size) or (right_matrix.shape[0] < toler_size):
                continue
            split_error = var_error_sum(left_matrix[:, -1]) + var_error_sum(right_matrix[:, -1])
            if split_error < least_square_error:
                least_square_error = split_error
                chosen_feature_index = feature_index
                chosen_feature_value = value
                find_split_flag = 1

    total_error = var_error_sum(y)
    if ((total_error - least_square_error) < toler_error) or (find_split_flag == 0):
        return None, np.mean(y)
    return chosen_feature_index, chosen_feature_value


def create_reg_tree(data_matrix, option):
    feature_index, feature_value = choose_feature(data_matrix, option)
    if feature_index == None:
        return feature_value
    reg_tree = {}
    reg_tree['feature_index'] = feature_index
    reg_tree['feature_value'] = feature_value

    left_subtree, right_subtree = split(data_matrix, feature_index, feature_value)
    reg_tree['left'] = create_reg_tree(left_subtree, option)
    reg_tree['right'] = create_reg_tree(right_subtree, option)
    return reg_tree


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right'])/2.0


def prune(tree, test_matrix):
    if test_matrix.shape[0] == 0:
        return get_mean(tree)

    if is_tree(tree['left']) or is_tree(tree['right']):
        left_set, right_set = split(test_matrix, tree['feature_index'], tree['feature_value'])
        if is_tree(tree['left']):
            tree['left'] = prune(tree['left'], left_set)
        if is_tree(tree['right']):
            tree['right'] = prune(tree['right'], right_set)

    if not is_tree(tree['left']) and not is_tree(tree['right']):
        left_set, right_set = split(test_matrix, tree['feature_index'], tree['feature_value'])
        split_error = sum(np.power(left_set[:, -1] - tree['left'], 2)) + \
                      sum(np.power(right_set[:, -1] - tree['right'], 2))
        merge_value = (tree['left'] + tree['right'])/2.0
        merge_error = sum(np.power(test_matrix[:, -1] - merge_value, 2))
        if merge_error <= split_error:
            print ('Merge')
            return merge_value
        else:
            return tree
    else:
        return tree

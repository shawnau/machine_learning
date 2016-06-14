import numpy as np
import operator
import copy


# Notice: the main difference between this and ID3 splitting function
# is that CART could choose the same feature in different nodes.
def split_continuous(data_matrix, feature_index, value):
    value_list = data_matrix[:, feature_index]

    # left_indices = (array([ 8,  9, 10, 13], dtype=int64), ), which is a tuple
    # while left_indices[0] = array([ 8,  9, 10, 13], dtype=int64), which is an array
    left_indices = np.where(value_list <= value)[0]
    left_matrix = data_matrix[left_indices]

    right_indices = np.where(value_list > value)[0]
    right_matrix = data_matrix[right_indices]
    return left_matrix, right_matrix


# This is similar to ID3 splitting, but it will only remove the splitted feature on the equal side,
# while keep splitted feature on the unequal side
def split_discrete(data_matrix, feature_index, value):
    equal_indices = np.where(data_matrix[:, feature_index] == value)[0]
    unequal_indices = np.where(data_matrix[:, feature_index] != value)[0]
    equal_matrix = data_matrix[equal_indices, :]
    eq_split_matrix = np.hstack((equal_matrix[:, :feature_index], equal_matrix[:, feature_index+1:]))
    unequal_matrix = data_matrix[unequal_indices, :]
    return eq_split_matrix, unequal_matrix


# ---------------------------------------Regression Tree----------------------------------------
# Only works on continuous feature values with continuous label y
def var_error_sum(data_list):
    return data_list.shape[0] * np.var(data_list)


def choose_feature_r(data_matrix, option):
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
            left_matrix, right_matrix = split_continuous(data_matrix, feature_index, value)
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
    feature_index, feature_value = choose_feature_r(data_matrix, option)
    if feature_index == None:
        return feature_value
    reg_tree = {}
    reg_tree['feature_index'] = feature_index
    reg_tree['feature_value'] = feature_value

    left_subtree, right_subtree = split_continuous(data_matrix, feature_index, feature_value)
    reg_tree['left'] = create_reg_tree(left_subtree, option)
    reg_tree['right'] = create_reg_tree(right_subtree, option)
    return reg_tree


# -----------------------------------Classification Tree--------------------------------------
# Only works on discrete feature values with discrete labels y
def gini(data_matrix):
    n = data_matrix.shape[0]
    class_count = {}
    for line_index in range(n):
        sample_class = data_matrix[line_index, -1]
        if sample_class not in class_count.keys():
            class_count[sample_class] = 0
        class_count[sample_class] += 1

    gini = 1.0
    for key in class_count:
        mle_prob = class_count[key]/float(n)
        gini -= mle_prob**2
    return gini


def splitted_gini(data_matrix, feature_index, value):
    left_matrix, right_matrix = split_discrete(data_matrix, feature_index, value)
    cond_gini = (left_matrix.shape[0]/float(data_matrix.shape[0])) * gini(left_matrix) + \
                (right_matrix.shape[0]/float(data_matrix.shape[0])) * gini(right_matrix)
    return cond_gini


def major_class(data_matrix):
    class_count = {}
    class_list = list(set(data_matrix[:, -1]))
    for class_name in class_list:
        class_count[class_name] = list(data_matrix[:, -1]).count(class_name)

    max_size = 0
    max_class = None
    for key in class_count:
        if class_count[key] > max_size:
            max_size = class_count[key]
            max_class = key
    return max_class


def choose_feature_c(data_matrix, option):
    toler_error = option[0]
    toler_size = option[1]
    m, n = data_matrix.shape
    y = data_matrix[:, -1]

    if list(y).count(y[0]) == len(y):
        return None, major_class(data_matrix)
    if data_matrix.shape[1] == 1:
        return None, major_class(data_matrix)

    largest_cond_gini = 0.0
    chosen_feature_index = -1
    chosen_feature_value = 0.0
    find_split_flag = 0
    for feature_index in range(n-1):
        for value in set(data_matrix[:, feature_index]):
            eq_matrix, uneq_matrix = split_discrete(data_matrix, feature_index, value)
            if (eq_matrix.shape[0] < toler_size) or (uneq_matrix.shape[0] < toler_size):
                continue
            split_gini = splitted_gini(data_matrix, feature_index, value)
            if split_gini > largest_cond_gini:
                largest_cond_gini = split_gini
                chosen_feature_index = feature_index
                chosen_feature_value = value
                find_split_flag = 1

    if (largest_cond_gini < toler_error) or (find_split_flag == 0):
        return None, major_class(data_matrix)
    return chosen_feature_index, chosen_feature_value


# option[0]:tolerance error, option[1]:tolerance size
def create_class_tree(data_matrix, feature_names, option):
    feature_index, feature_value = choose_feature_c(data_matrix, option)
    if feature_index == None:
        return feature_value
    class_tree = {}
    class_tree['feature_name'] = feature_names[feature_index]
    class_tree['feature_value'] = feature_value

    equal_subtree, unequal_subtree = split_discrete(data_matrix, feature_index, feature_value)

    equal_feature_names = copy.deepcopy(feature_names)
    del(equal_feature_names[feature_index])
    class_tree['equal'] = create_class_tree(equal_subtree, equal_feature_names, option)
    class_tree['unequal'] = create_class_tree(unequal_subtree, feature_names, option)
    return class_tree


# ---------------------------------------Tree prune-------------------------------------------
# Only works on regression tree for now
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
        left_set, right_set = split_continuous(test_matrix, tree['feature_index'], tree['feature_value'])
        if is_tree(tree['left']):
            tree['left'] = prune(tree['left'], left_set)
        if is_tree(tree['right']):
            tree['right'] = prune(tree['right'], right_set)

    if not is_tree(tree['left']) and not is_tree(tree['right']):
        left_set, right_set = split_continuous(test_matrix, tree['feature_index'], tree['feature_value'])
        split_error = sum(np.power(left_set[:, -1] - tree['left'], 2)) + \
                      sum(np.power(right_set[:, -1] - tree['right'], 2))
        merge_value = (tree['left'] + tree['right'])/2.0
        merge_error = sum(np.power(test_matrix[:, -1] - merge_value, 2))
        if merge_error <= split_error:
            # print ('Merge')
            return merge_value
        else:
            return tree
    else:
        return tree


# ------------------------------------Tree Prediction---------------------------------------------
# Notice: if feature is 1-D vector will error
def regression_predict(input_tree, input_x):
    if type(input_tree).__name__ == 'dict':
        left_subtree = input_tree['left']
        right_subtree = input_tree['right']
        # input_x[input_tree['feature_index']] for 2D+ feature
        if input_x <= input_tree['feature_value']:
            predicted_class = regression_predict(left_subtree, input_x)
        else:
            predicted_class = regression_predict(right_subtree, input_x)
    else:
        predicted_class = input_tree
    return predicted_class


def class_predict(input_tree, feature_names, input_x):
    if type(input_tree).__name__ == 'dict':
        unequal_subtree = input_tree['unequal']
        equal_subtree = input_tree['equal']
        if input_x[feature_names.index(input_tree['feature_name'])] == input_tree['feature_value']:
            predicted_class = class_predict(equal_subtree, feature_names, input_x)
        else:
            predicted_class = class_predict(unequal_subtree, feature_names, input_x)
    else:
        predicted_class = input_tree
    return predicted_class

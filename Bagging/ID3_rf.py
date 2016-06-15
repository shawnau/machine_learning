import numpy as np
from math import log
import random
import operator
import copy


def calc_entropy(data_matrix):
    sample_num = data_matrix.shape[0]
    class_count = {}
    for i in range(sample_num):
        sample_vector = data_matrix[i, :]
        sample_class = sample_vector[-1]
        if sample_class not in class_count.keys():
            class_count[sample_class] = 0
        class_count[sample_class] += 1

    entropy = 0.0
    for key in class_count:
        mle_prob = class_count[key]/float(sample_num)
        entropy -= mle_prob * log(mle_prob, 2)
    return entropy


def split(data_matrix, feature_index, value):
    line_indices = np.where(data_matrix[:, feature_index] == value)[0]
    sub_matrix = data_matrix[line_indices]
    split_matrix = np.hstack((sub_matrix[:, :feature_index], sub_matrix[:, feature_index+1:]))
    return split_matrix


def calc_cond_entropy(data_matrix, feature_index):
    value_list = data_matrix[:, feature_index]
    cond_entropy = 0.0
    for value in set(value_list):
        reduced_matrix = split(data_matrix, feature_index, value)
        prob = reduced_matrix.shape[0]/float(data_matrix.shape[0])
        cond_entropy += prob * calc_entropy(reduced_matrix)
    return cond_entropy


# RF_size is the size of features chosen when using random forest
def choose_feature(data_matrix, rf):
    feature_num = data_matrix.shape[1] - 1
    chosen_info_gain = 0.0
    chosen_feature_index = -1
    if rf == 0:
        feature_indices = range(feature_num)
    elif rf == 1:
        feature_size = int(log(feature_num, 2))
        feature_indices = np.random.permutation(range(feature_num))[:feature_size]
    else:
        raise NameError('RF switch not recognized')

    for i in feature_indices:
        info_gain = calc_entropy(data_matrix) - calc_cond_entropy(data_matrix, i)
        if info_gain > chosen_info_gain:
            chosen_info_gain = info_gain
            chosen_feature_index = i
    return chosen_feature_index, chosen_info_gain


def major_class(class_vector):
    class_count = {}
    m = class_vector.shape[0]
    for i in range(m):
        vote = class_vector[i]
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def tree_creation(data_matrix, feature_names, rf, tolerance=0.0):
    class_vector = data_matrix[:, -1]
    if np.where(class_vector == class_vector[0])[0].shape[0] == class_vector.shape[0]:
        return class_vector[0]
    if data_matrix.shape[1] == 1:
        return major_class(class_vector)

    chosen_feature_index, chosen_info_gain = choose_feature(data_matrix, rf)
    if chosen_info_gain < tolerance:
        return major_class(class_vector)
    chosen_feature_name = feature_names[chosen_feature_index]
    decision_tree = {chosen_feature_name: {}}

    value_list = data_matrix[:, chosen_feature_index]
    unique_values = set(value_list)

    local_feature_names = copy.deepcopy(feature_names)
    del(local_feature_names[chosen_feature_index])
    for value in unique_values:
        decision_tree[chosen_feature_name][value] = \
            tree_creation(split(data_matrix, chosen_feature_index, value), local_feature_names, tolerance)
    return decision_tree


def tree_prediction(input_tree, feature_names, input_x):
    if type(input_tree).__name__ == 'dict':
        feature_value = input_x[feature_names.index(input_tree.keys()[0])]
        # randomly get KeyError here, fixed using randomly picking one value in the tree
        # if the value doesn't exist in the tree when using random forest training samples,
        # which would lost some values in some particular training samples
        if feature_value in input_tree[input_tree.keys()[0]].keys():
            sub_tree = input_tree[input_tree.keys()[0]][feature_value]
        else:
            random_key = random.sample(input_tree[input_tree.keys()[0]], 1)[0]
            sub_tree = input_tree[input_tree.keys()[0]][random_key]
        return tree_prediction(sub_tree, feature_names, input_x)  # this return is necessary
    else:
        predicted_class = input_tree
        return predicted_class

from math import log
import operator


def calc_entropy(data_set):
    sample_num = len(data_set)
    class_count = {}
    for sample_vector in data_set:
        sample_class = sample_vector[-1]
        if sample_class not in class_count.keys():
            class_count[sample_class] = 0
        class_count[sample_class] += 1

    entropy = 0.0
    for key in class_count:
        mle_prob = class_count[key]/float(sample_num)
        entropy -= mle_prob * log(mle_prob, 2)
    return entropy


def split(data_set, feature_index, value):
    reduced_data = []
    for feature_vector in data_set:
        if feature_vector[feature_index] == value:
            reduced_vector = feature_vector[:feature_index]
            reduced_vector.extend(feature_vector[feature_index + 1:])
            reduced_data.append(reduced_vector)
    return reduced_data


def calc_cond_entropy(data_set, feature_index):
    value_list = [vector[feature_index] for vector in data_set]
    cond_entropy = 0.0
    for value in set(value_list):
        reduced_data = split(data_set, feature_index, value)
        prob = len(reduced_data)/float(len(data_set))
        cond_entropy += prob * calc_entropy(reduced_data)
    return cond_entropy


def choose_feature(data_set):
    feature_num = len(data_set[0]) - 1
    chosen_info_gain = 0.0
    chosen_feature_index = -1
    for i in range(feature_num):
        info_gain = calc_entropy(data_set) - calc_cond_entropy(data_set, i)
        if info_gain > chosen_info_gain:
            chosen_info_gain = info_gain
            chosen_feature_index = i
    return chosen_feature_index, chosen_info_gain


def major_class(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def tree_creation(data_set, feature_list, tolerance=0.0):
    class_list = [vector[-1] for vector in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return major_class(class_list)

    chosen_feature_index, chosen_info_gain = choose_feature(data_set)
    if chosen_info_gain < tolerance:
        return major_class(class_list)
    chosen_feature_name = feature_list[chosen_feature_index]
    decision_tree = {chosen_feature_name: {}}

    value_list = [vector[chosen_feature_index] for vector in data_set]
    unique_values = set(value_list)

    del(feature_list[chosen_feature_index])
    for value in unique_values:
        decision_tree[chosen_feature_name][value] = \
            tree_creation(split(data_set, chosen_feature_index, value), feature_list, tolerance)
    return decision_tree

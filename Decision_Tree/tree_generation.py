from math import log


def calc_entropy(data_set):
    sample_num = len(data_set)
    class_count = {}
    for sample_vector in data_set:
        sample_class = sample_vector[-1]
        if sample_class not in class_count:
            class_count[sample_class] = 0
        class_count[sample_class] += 1

    entropy = 0.0
    for key in class_count:
        mle_prob = class_count[key]/sample_num
        entropy -= mle_prob * log(mle_prob, 2)
    return entropy


def calc_cond_entropy(data_set, feature_index):
    value_list = [vector[feature_index] for vector in data_set]
    unique_value = set(value_list)
    cond_entropy = 0.0
    for value in unique_value:
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
    return chosen_feature_index


def split(data_set, feature_index, value):
    reduced_data = []
    for feature_vector in data_set:
        if feature_vector[feature_index] == value:
            reduced_vector = feature_vector[:feature_index]
            reduced_vector.extend(feature_vector[feature_index + 1:])
            reduced_data.append(reduced_vector)
    return reduced_data


def tree_creation(data_set, labels):
    pass

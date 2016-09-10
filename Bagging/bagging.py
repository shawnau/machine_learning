import numpy as np
import ID3_rf as model
import operator


def bagging(data_matrix, feature_names, rf=0, iteration=40):
    """
    :param data_matrix: one sample vector each row, the last element is the class label y
    :param feature_names: a list consist of each feature's name
    :param rf: flag for using randomforest
    :param iteration: default:40
    :return: a list of trees implemented through diction
    """
    m, n = np.shape(data_matrix)
    model_list = []
    for i in range(iteration):
        sample_indices = np.random.random_integers(0, m-1, (m,))
        sample_matrix = data_matrix[sample_indices]
        base = model.tree_creation(sample_matrix, feature_names, rf)
        model_list.append(base)
    return model_list


def bagging_classify(input_x, feature_names, model_list):
    """
    :param input_x: input column vector
    :param feature_names: a list of feature names
    :return: a list of trees implemented through diction 
    """
    vote = {}
    for i in range(len(model_list)):
        base_prediction = model.tree_prediction(model_list[i], feature_names, input_x)
        if base_prediction not in vote.keys():
            vote[base_prediction] = 0
        vote[base_prediction] += 1
    return max(vote.iteritems(), key=operator.itemgetter(1))[0]

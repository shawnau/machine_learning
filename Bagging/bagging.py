import numpy as np
import ID3_rf as model
import operator


def bagging(data_matrix, feature_names, rf=0, iteration=40):
    m, n = np.shape(data_matrix)
    model_list = []
    for i in range(iteration):
        sample_indices = np.random.random_integers(0, m-1, (m,))
        sample_matrix = data_matrix[sample_indices]
        base = model.tree_creation(sample_matrix, feature_names, rf)
        model_list.append(base)
    return model_list


def bagging_classify(input_x, feature_names, model_list):
    vote = {}
    for i in range(len(model_list)):
        base_prediction = model.tree_prediction(model_list[i], feature_names, input_x)
        if base_prediction not in vote.keys():
            vote[base_prediction] = 0
        vote[base_prediction] += 1
    return max(vote.iteritems(), key=operator.itemgetter(1))[0]

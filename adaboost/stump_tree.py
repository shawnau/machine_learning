import numpy as np


def stump_classifier(data_matrix, feature_index, threshold, rule='lt'):
    results = np.ones((data_matrix.shape[0], 1))
    feature_values = data_matrix[:, feature_index]
    if rule == 'lt':
        results[np.where(feature_values <= threshold)[0]] = -1.0
    else:
        results[np.where(feature_values > threshold)[0]] = -1.0
    return results


# desision tree with only 2 leaf nodes, works on continuous feature value
# and 2-value prediction only
def create_stump(data_matrix, class_vector, data_weights, step_number=10):
    m, n = np.shape(data_matrix)
    stump = {}
    return_prediction = np.zeros((m, 1))
    min_error = np.inf

    for i in range(n):
        min_value = data_matrix[:, i].min()
        max_value = data_matrix[:, i].max()
        step_size = (max_value - min_value) / float(step_number)
        for j in range(-1, step_number + 1):
            for rule in ['lt', 'gt']:
                threshold = min_value + j * step_size
                prediction = stump_classifier(data_matrix, i, threshold, rule)
                is_error = np.ones((m, 1))
                is_error[prediction == class_vector] = 0
                weighted_error = np.dot(data_weights.T, is_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    return_prediction = prediction.copy()
                    stump['feature_index'] = i
                    stump['threshold'] = threshold
                    stump['rule'] = rule
    return stump, return_prediction, min_error

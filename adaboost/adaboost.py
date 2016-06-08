import math
import numpy as np
import stump_tree as st


def adaboost(data_matrix, class_vector, iteration=40):
    m = data_matrix.shape[0]
    data_weights = np.ones((m, 1)) / m  # elements are float
    model_weights = np.zeros((iteration, 1))
    model_list = []
    for i in range(iteration):
        stump, predictions, weighted_error = st.create_stump(data_matrix, class_vector, data_weights)
        model_list.append(stump)
        model_weights[i] = 0.5 * math.log((1.0 - weighted_error) / max(weighted_error, 1e-16))

        data_weights = data_weights * np.exp(-1.0 * model_weights[i] * class_vector * predictions)
        data_weights = data_weights / np.sum(data_weights)
    return model_weights, model_list


def adaboost_test(data_matrix, class_vector):
    model_weights, model_list = adaboost(data_matrix, class_vector)
    models_output = np.zeros((data_matrix.shape[0], 1))
    for i in range(len(model_list)):
        model_prediction = st.stump_classifier(data_matrix,
                                               model_list[i]['feature_index'],
                                               model_list[i]['threshold'],
                                               model_list[i]['rule'])
        models_output += model_weights[i] * model_prediction

    error_rate = np.where((np.sign(models_output) == class_vector) == False)[0].shape[0] / float(data_matrix.shape[0])
    print('model weights: ', model_weights, 'models_output: ', models_output, 'error rate: ', error_rate)


def adaboost_classify(input_matrix, model_weights, model_list):
    m = input_matrix.shape[0]
    models_output = np.zeros((m, 1))
    for i in range(len(model_list)):
        model_prediction = st.stump_classifier(input_matrix,
                                               model_list[i]['feature_index'],
                                               model_list[i]['threshold'],
                                               model_list[i]['rule'])
        models_output += model_weights[i] * model_prediction
    return np.sign(models_output)

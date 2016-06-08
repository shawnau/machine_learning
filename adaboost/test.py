import numpy as np
import adaboost as ab


def create_simple_data():
    create_matrix = np.array([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]])
    create_vector = np.array([[1.0], [1.0], [-1.0], [-1.0], [1.0]])
    return create_matrix, create_vector


data_matrix, class_vector = create_simple_data()
model_weights, model_list = ab.adaboost(data_matrix, class_vector)
ab.adaboost_test(data_matrix, class_vector)
results = ab.adaboost_classify(data_matrix, model_weights, model_list)

print('class labels: ', class_vector)
print('predicted results: ', results)

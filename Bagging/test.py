import bagging as bg
import numpy as np


def create_test_data():
    data_set = [[0, 0, 0, 0, 'n'],
                [0, 0, 0, 1, 'n'],
                [0, 1, 0, 1, 'y'],
                [0, 1, 1, 0, 'y'],
                [0, 0, 0, 0, 'n'],
                [1, 0, 0, 0, 'n'],
                [1, 0, 0, 1, 'n'],
                [1, 1, 1, 1, 'y'],
                [1, 0, 1, 2, 'y'],
                [1, 0, 1, 2, 'y'],
                [2, 0, 1, 2, 'y'],
                [2, 0, 1, 1, 'y'],
                [2, 1, 0, 1, 'y'],
                [2, 1, 0, 2, 'y'],
                [2, 0, 0, 0, 'n']]
    feature_names = ['age', 'have job', 'have house', 'credit situation']
    return data_set, feature_names


data_list, feature_list = create_test_data()
data_matrix = np.array(data_list)
model_list = bg.bagging(data_matrix, feature_list, (0.0, 9))
prediction = bg.bagging_classify([2, 1, 0, 2], feature_list, model_list)

print(prediction)
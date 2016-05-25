import tree_creation as tc


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
    feature_list = ['age', 'have job', 'have house', 'credit situation']
    return data_set, feature_list


data, feature = create_test_data()
decision_tree = tc.tree_creation(data, feature)
print (decision_tree)
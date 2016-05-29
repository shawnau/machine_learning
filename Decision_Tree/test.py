import ID3 as id_three
import CART as cart
import test_toolkit as tt


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


# --------------Sample-----------------------
# data, feature = create_test_data()
# decision_tree = id_three.tree_creation(data, feature)
# print (decision_tree)

# data_matrix = tt.load_data('test_data/ex00.txt')
# decision_tree = cart.create_reg_tree(data_matrix, (1.0, 4))
# print (decision_tree)

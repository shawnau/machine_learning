import ID3 as id_three
import CART as cart
import test_toolkit as tt
import numpy as np
import copy


print('--------------ID3 Sample-----------------------------------')
data_matrix = tt.load_data('test_data/housingdata.txt')
feature_list = ['age', 'have job', 'have house', 'credit situation']
id3_decision_tree = id_three.tree_creation(data_matrix, feature_list)
print(id3_decision_tree)

prediction = id_three.tree_prediction(id3_decision_tree, feature_list, [2, 0, 0, 0])
print(prediction)

print('--------------CART Regression Sample-----------------------')
data_matrix = tt.load_data('test_data/ex2test.txt')
train_matrix, cv_matrix, test_matrix = tt.split_data(data_matrix, (0.45, 0.1))

decision_tree = cart.create_reg_tree(train_matrix, (0, 1))
pruned_tree = cart.prune(decision_tree, test_matrix)
print(cart.regression_predict(pruned_tree, 0.46))

print('-------------CART Classification Sample--------------------')
data_matrix = tt.load_data('test_data/housingdata.txt')
feature_list = ['age', 'have job', 'have house', 'credit situation']
decision_tree = cart.create_class_tree(data_matrix, feature_list, (0.0, 9))
 
m, n = np.shape(data_matrix)
correct_count = 0
for i in range(m):
    predicted_class = cart.class_predict(decision_tree, feature_list, data_matrix[i, :n-1])
    if predicted_class == data_matrix[i, n-1]:
        correct_count += 1
print('Sample size: ' + str(m) + ', classification correct prediction: ' + str(correct_count))


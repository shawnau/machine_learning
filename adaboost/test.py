import adaboost as ab
import test_toolkit as tt
import plot as plot

data_matrix = tt.load_data('test_data/horseColicTest2.txt')
train_matrix, cv_matrix, test_matrix = tt.split_data(data_matrix, (0.8, 0.0))
train_matrix_x, class_vector = tt.separate_x_y(train_matrix)
model_weights, model_list = ab.adaboost(train_matrix_x, class_vector)
ab.adaboost_test(train_matrix_x, class_vector)

test_matrix_x, test_matrix_class = tt.separate_x_y(test_matrix)
results = ab.adaboost_classify(test_matrix_x, model_weights, model_list)

print('class labels: ', test_matrix_class)
print('predicted results: ', results)
plot.plot_roc(results, test_matrix_class)

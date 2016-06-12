import adaboost as ab
import test_toolkit as tt
import plot as plot

data_matrix, class_vector = tt.separate_x_y(tt.load_data('test_data/horseColicTest2.txt'))
model_weights, model_list = ab.adaboost(data_matrix, class_vector)
ab.adaboost_test(data_matrix, class_vector)
results = ab.adaboost_classify(data_matrix, model_weights, model_list)

print('class labels: ', class_vector)
print('predicted results: ', results)
plot.plot_roc(results, class_vector)
